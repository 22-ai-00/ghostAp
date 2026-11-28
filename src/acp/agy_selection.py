"""AGY ACP and Orchestrator CLI model and thought-level selection helpers."""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .model_selection import (
    CODEX_REASONING_EFFORTS,
    compose_slash_model_effort,
    split_slash_model_effort,
)

AGY_ORCHESTRATOR_MODE_ID = "gemini-pro-orchestrator"
AGY_DEFAULT_MODEL_ID = AGY_ORCHESTRATOR_MODE_ID
AGY_ORCHESTRATOR_PLANNER_MODELS: tuple[str, ...] = (
    "gemini-4-neon",
    "gemini-4-carbon",
    "gemini-4-argon",
    "claude-opus-5-5",
)
AGY_ORCHESTRATOR_EXECUTOR_MODELS: tuple[str, ...] = (
    "gemini-4-hydrogen",
    "gemini-4-neon",
    "claude-sonnet-5-5",
)
AGY_PLANNER_NATIVE_ORCHESTRATOR_IDS: dict[str, str] = {
    "gemini-4-neon": "gemini-pro-orchestrator-neon",
    "gemini-4-carbon": "gemini-pro-orchestrator",
    "gemini-4-argon": "gemini-pro-orchestrator-argon",
}
AGY_MODEL_CONFIG_ID = "model"
AGY_THOUGHT_LEVEL_CONFIG_ID = "thought_level"
AGY_EFFORT_CONFIG_IDS: tuple[str, ...] = (
    AGY_THOUGHT_LEVEL_CONFIG_ID,
    "reasoning_effort",
    "effort",
)
AGY_REASONING_EFFORTS = CODEX_REASONING_EFFORTS
AGY_ORCHESTRATOR_DEFAULT_PROFILE = "default"
_AGY_ROLE_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

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

AGY_KNOWN_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gemini-4-neon": "Gemini 4 Neon",
    "gemini-4-carbon": "Gemini 4 Carbon",
    "gemini-4-argon": "Gemini 4 Argon",
    "gemini-4-hydrogen": "Gemini 4 Hydrogen",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "gpt-oss-120b-medium": "GPT-OSS 120B (Medium)",
    "gemini-pro-orchestrator-neon": "Orchestrator (Neon)",
    "gemini-pro-orchestrator": "Orchestrator (Carbon)",
    "gemini-pro-orchestrator-argon": "Orchestrator (Argon)",
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
    planner_model: str | None = None
    executor_model: str | None = None


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


def is_valid_agy_planner_model(model_id: str | None) -> bool:
    """Return True if *model_id* is one of the 4 allowed AGY Orchestrator Planner models."""
    return str(model_id or "").strip() in AGY_ORCHESTRATOR_PLANNER_MODELS


def is_valid_agy_executor_model(model_id: str | None) -> bool:
    """Return True if *model_id* is one of the 3 allowed AGY Orchestrator Executor models."""
    return str(model_id or "").strip() in AGY_ORCHESTRATOR_EXECUTOR_MODELS


def _filter_agy_role_models(
    allowlist: tuple[str, ...],
    available_names: Sequence[str] = (),
) -> tuple[str, ...]:
    clean_available = {
        str(name or "").strip()
        for name in (available_names or ())
        if str(name or "").strip()
    }
    if not clean_available:
        return allowlist
    return tuple(model for model in allowlist if model in clean_available)


def filter_agy_planner_models(available_names: Sequence[str] = ()) -> tuple[str, ...]:
    """Filter *available_names* to allowed Planner models in canonical order."""
    return _filter_agy_role_models(AGY_ORCHESTRATOR_PLANNER_MODELS, available_names)


def filter_agy_executor_models(available_names: Sequence[str] = ()) -> tuple[str, ...]:
    """Filter *available_names* to allowed Executor models in canonical order."""
    return _filter_agy_role_models(AGY_ORCHESTRATOR_EXECUTOR_MODELS, available_names)


def collapse_agy_orchestrator_group_names(model_names: Sequence[str]) -> tuple[str, ...]:
    """Collapse all Orchestrator entries into one unified Orchestrator entry at first position."""
    cleaned = [str(name or "").strip() for name in (model_names or ()) if str(name or "").strip()]
    orch_ids = [name.split("/", 1)[0].strip() for name in cleaned if is_agy_orchestrator_model(name)]
    if not orch_ids:
        return tuple(dict.fromkeys(cleaned))
    preferred = (
        AGY_ORCHESTRATOR_MODE_ID
        if AGY_ORCHESTRATOR_MODE_ID in orch_ids
        else orch_ids[0]
    )
    collapsed: list[str] = []
    inserted = False
    for name in cleaned:
        if is_agy_orchestrator_model(name):
            if not inserted:
                collapsed.append(preferred)
                inserted = True
            continue
        if name not in collapsed:
            collapsed.append(name)
    return tuple(collapsed)


def canonical_agy_model_group(
    model_id: str | None,
    group_names: Sequence[str] = (),
) -> str:
    """Map any AGY selection (regular or any Orchestrator) to its collapsed group entry."""
    raw = str(model_id or "").strip()
    if not raw:
        return ""
    groups = tuple(str(name or "").strip() for name in (group_names or ()) if str(name or "").strip())
    if is_agy_orchestrator_model(raw):
        orch_groups = [name for name in groups if is_agy_orchestrator_model(name)]
        if AGY_ORCHESTRATOR_MODE_ID in orch_groups:
            return AGY_ORCHESTRATOR_MODE_ID
        if orch_groups:
            return orch_groups[0]
        return AGY_ORCHESTRATOR_MODE_ID
    if raw in groups:
        return raw
    base = raw.split("/", 1)[0].strip()
    if base in groups:
        return base
    if not groups:
        try:
            return parse_agy_model_selection(raw).model_id
        except ValueError:
            return base
    return raw


def resolve_native_orchestrator_preset(
    planner_model: str | None,
    available_names: Sequence[str] = (),
) -> str | None:
    """Return the native ACP Orchestrator preset ID for *planner_model*, if supported & present."""
    clean_planner = str(planner_model or "").strip()
    preset = AGY_PLANNER_NATIVE_ORCHESTRATOR_IDS.get(clean_planner)
    if not preset:
        return None
    clean_available = {
        str(name or "").strip()
        for name in (available_names or ())
        if str(name or "").strip()
    }
    if clean_available and preset not in clean_available:
        return None
    return preset


def format_agy_role_option_label(model_id: str) -> str:
    """Return the friendly display label for an AGY Planner or Executor model option."""
    clean = str(model_id or "").strip()
    if not clean:
        return ""
    with _LABEL_LOCK:
        friendly = _LIVE_BASE_DISPLAY_NAMES.get(clean)
    return friendly or AGY_KNOWN_MODEL_DISPLAY_NAMES.get(clean) or clean


def format_agy_group_option_label(model_id: str) -> str:
    """Format a top-level AGY model-group dropdown option label (<= 44 chars for regular models)."""
    clean = str(model_id or "").strip()
    if not clean:
        return ""
    if is_agy_orchestrator_model(clean):
        return "🪐 Orchestrator（双阶段规划+执行）"
    if clean.lower() == "default":
        return "default（后端托管）"
    friendly = format_agy_role_option_label(clean)
    label = f"{friendly} · {clean}" if friendly and friendly != clean else clean
    if len(label) <= 44:
        return label
    return f"{label[:43].rstrip()}…"


def format_agy_orchestrator_summary(selection: str | None) -> str:
    """Format a concise summary for the active AGY Orchestrator selection."""
    raw = str(selection or "").strip()
    if not raw:
        return ""
    try:
        parsed = parse_agy_model_selection(raw)
    except ValueError:
        return raw
    spec = AGY_ORCHESTRATOR_SPECS.get(parsed.model_id)
    if spec is None:
        return format_agy_role_label(parsed.model_id, parsed.thought_level)
    if not requires_agy_cli_orchestrator_session(raw):
        return (
            f"原生 ACP 预设 · Planner {spec.planner_display_base}"
            f" + Executor {spec.executor_display_base}（支持 /swarm）"
        )
    planner_model = parsed.planner_model or spec.planner_model_id
    executor_model = parsed.executor_model or spec.executor_model_id
    planner_label = format_agy_role_label(
        planner_model,
        parsed.planner_effort,
        display_base=(spec.planner_display_base if planner_model == spec.planner_model_id else None),
    )
    executor_label = format_agy_role_label(
        executor_model,
        parsed.executor_effort,
        display_base=(spec.executor_display_base if executor_model == spec.executor_model_id else None),
    )
    return f"Planner {planner_label} → Executor {executor_label}"


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
        base_label = (
            str(display_base or "").strip()
            or AGY_KNOWN_MODEL_DISPLAY_NAMES.get(norm_id)
            or norm_id
        )
    effort_label = AGY_EFFORT_DISPLAY_LABELS.get(norm_effort, norm_effort.title())
    if base_label.endswith(f"({effort_label})"):
        return base_label
    return f"{base_label} ({effort_label})"


def format_agy_role_label(
    model_id: str,
    effort: str | None,
    *,
    display_base: str | None = None,
) -> str:
    """Resolve the display label for an Orchestrator Planner or Executor role."""
    norm_id = str(model_id or "").strip()
    if not norm_id:
        raise ValueError(f"invalid AGY model for role label: {model_id!r}")
    if effort:
        return format_agy_model_effort_label(
            norm_id,
            effort,
            display_base=display_base,
        )
    with _LABEL_LOCK:
        base_label = _LIVE_BASE_DISPLAY_NAMES.get(norm_id)
    return (
        base_label
        or str(display_base or "").strip()
        or AGY_KNOWN_MODEL_DISPLAY_NAMES.get(norm_id)
        or norm_id
    )


def _is_valid_agy_role_model_slug(slug: str) -> bool:
    clean = str(slug or "").strip()
    if (
        not clean
        or clean in AGY_ORCHESTRATOR_SPECS
        or not _AGY_ROLE_MODEL_RE.match(clean)
    ):
        return False
    return any(ch == "-" or ch.isdigit() for ch in clean)


def _parse_agy_role_token(
    token: str,
    default_model_id: str,
) -> tuple[str, str | None]:
    raw = str(token or "").strip()
    if not raw:
        raise ValueError("empty AGY orchestrator role token")
    lowered = raw.lower()
    if lowered in AGY_REASONING_EFFORTS:
        return default_model_id, lowered
    if ":" in raw:
        if raw.count(":") != 1:
            raise ValueError(f"invalid AGY orchestrator role token: {token}")
        model_part, eff_part = raw.split(":", 1)
        model_id = model_part.strip()
        effort = eff_part.strip().lower()
        if not _is_valid_agy_role_model_slug(model_id) or effort not in AGY_REASONING_EFFORTS:
            raise ValueError(f"invalid AGY orchestrator role token: {token}")
        return model_id, effort
    if not _is_valid_agy_role_model_slug(raw):
        raise ValueError(f"invalid AGY orchestrator role token: {token}")
    return raw, None


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
    *,
    planner_model: str | None = None,
    executor_model: str | None = None,
) -> str:
    """Compose GhostAP's durable AGY Orchestrator selection (`orch`, `orch/p/e`, or `orch/pm:pe/em:ee`)."""
    spec = get_agy_orchestrator_spec(model_id)
    if spec is None or "/" in str(model_id or ""):
        raise ValueError(f"invalid AGY orchestrator model: {model_id}")
    raw_pm = str(planner_model or "").strip()
    raw_em = str(executor_model or "").strip()
    p_model = spec.planner_model_id if raw_pm in {"", AGY_ORCHESTRATOR_DEFAULT_PROFILE} else raw_pm
    e_model = spec.executor_model_id if raw_em in {"", AGY_ORCHESTRATOR_DEFAULT_PROFILE} else raw_em
    if not is_valid_agy_planner_model(p_model):
        raise ValueError(f"invalid AGY planner model: {planner_model}")
    if not is_valid_agy_executor_model(e_model):
        raise ValueError(f"invalid AGY executor model: {executor_model}")

    norm_planner = str(planner_effort or "").strip().lower()
    norm_executor = str(executor_effort or "").strip().lower()
    if norm_planner == AGY_ORCHESTRATOR_DEFAULT_PROFILE:
        norm_planner = ""
    if norm_executor == AGY_ORCHESTRATOR_DEFAULT_PROFILE:
        norm_executor = ""

    if not norm_planner and not norm_executor:
        if e_model == "gemini-4-hydrogen" and p_model in AGY_PLANNER_NATIVE_ORCHESTRATOR_IDS:
            return (
                spec.model_id
                if p_model == spec.planner_model_id
                else AGY_PLANNER_NATIVE_ORCHESTRATOR_IDS[p_model]
            )
        raise ValueError(
            f"AGY orchestrator native default is unavailable for planner={p_model!r}, executor={e_model!r}"
        )

    if p_model == spec.planner_model_id and e_model == spec.executor_model_id:
        if norm_planner not in AGY_REASONING_EFFORTS:
            raise ValueError(f"invalid AGY planner effort: {planner_effort}")
        if not norm_executor:
            norm_executor = norm_planner
        elif norm_executor not in AGY_REASONING_EFFORTS:
            raise ValueError(f"invalid AGY executor effort: {executor_effort}")
        return f"{spec.model_id}/{norm_planner}/{norm_executor}"

    if norm_planner not in AGY_REASONING_EFFORTS:
        raise ValueError(f"invalid AGY planner effort: {planner_effort}")
    if norm_executor not in AGY_REASONING_EFFORTS:
        raise ValueError(f"invalid AGY executor effort: {executor_effort}")
    return f"{spec.model_id}/{p_model}:{norm_planner}/{e_model}:{norm_executor}"


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
            spec = AGY_ORCHESTRATOR_SPECS[base]
            return AGYParsedSelection(
                model_id=base,
                planner_effort=eff,
                executor_effort=eff,
                planner_model=spec.planner_model_id,
                executor_model=spec.executor_model_id,
            )
        return AGYParsedSelection(model_id=base, thought_level=eff)
    if len(parts) == 3:
        base = parts[0].strip()
        if base not in AGY_ORCHESTRATOR_SPECS:
            raise ValueError(f"invalid AGY model selection: {selection}")
        spec = AGY_ORCHESTRATOR_SPECS[base]
        p_model, p_eff = _parse_agy_role_token(parts[1], spec.planner_model_id)
        e_model, e_eff = _parse_agy_role_token(parts[2], spec.executor_model_id)
        if (
            not is_valid_agy_planner_model(p_model)
            or not is_valid_agy_executor_model(e_model)
            or p_eff not in AGY_REASONING_EFFORTS
            or e_eff not in AGY_REASONING_EFFORTS
        ):
            raise ValueError(f"invalid AGY model selection: {selection}")
        return AGYParsedSelection(
            model_id=base,
            planner_effort=p_eff,
            executor_effort=e_eff,
            planner_model=p_model,
            executor_model=e_model,
        )
    if len(parts) == 5:
        base = parts[0].strip()
        p_model = parts[1].strip()
        p_eff_raw = parts[2].strip().lower()
        e_model = parts[3].strip()
        e_eff_raw = parts[4].strip().lower()
        if (
            base not in AGY_ORCHESTRATOR_SPECS
            or not is_valid_agy_planner_model(p_model)
            or not is_valid_agy_executor_model(e_model)
            or p_eff_raw not in AGY_REASONING_EFFORTS
            or e_eff_raw not in AGY_REASONING_EFFORTS
        ):
            raise ValueError(f"invalid AGY model selection: {selection}")
        return AGYParsedSelection(
            model_id=base,
            planner_effort=p_eff_raw,
            executor_effort=e_eff_raw,
            planner_model=p_model,
            executor_model=e_model,
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
    """Return True when *selection* is an Orchestrator with custom Planner/Executor settings."""
    raw = str(selection or "").strip()
    if not raw or "/" not in raw:
        return False
    try:
        parsed = parse_agy_model_selection(raw)
    except ValueError:
        return False
    return bool(
        parsed.model_id in AGY_ORCHESTRATOR_SPECS
        and (
            parsed.planner_effort is not None
            or parsed.executor_effort is not None
            or parsed.planner_model is not None
            or parsed.executor_model is not None
        )
    )


def resolve_agy_orchestrator_settings(
    selection: str | None,
) -> tuple[str, dict[str, Any]]:
    """Build `(orchestrator_model_id, settings_dict)` for `agy --settings`."""
    parsed = parse_agy_model_selection(selection)
    spec = AGY_ORCHESTRATOR_SPECS.get(parsed.model_id)
    if spec is None or not requires_agy_cli_orchestrator_session(selection):
        raise ValueError(f"selection is not an AGY orchestrator custom effort: {selection}")
    planner_model = parsed.planner_model or spec.planner_model_id
    executor_model = parsed.executor_model or spec.executor_model_id
    planner_label = format_agy_role_label(
        planner_model,
        parsed.planner_effort,
        display_base=(spec.planner_display_base if planner_model == spec.planner_model_id else None),
    )
    executor_label = format_agy_role_label(
        executor_model,
        parsed.executor_effort,
        display_base=(spec.executor_display_base if executor_model == spec.executor_model_id else None),
    )
    return spec.model_id, {
        "orchestrator": {
            "planner": planner_label,
            "executor": executor_label,
        }
    }


def is_valid_agy_catalog_selection(
    models: Sequence[Any],
    selection: str | None,
) -> bool:
    """Return True when *selection* is valid against discovered AGY ACP options."""
    raw = str(selection or "").strip()
    if not raw:
        return False
    try:
        parsed = parse_agy_model_selection(raw)
    except ValueError:
        return False
    by_name = {
        str(getattr(opt, "name", "") or "").strip(): opt
        for opt in (models or ())
        if str(getattr(opt, "name", "") or "").strip()
    }
    root_opt = by_name.get(parsed.model_id)
    if root_opt is None:
        return False
    if parsed.model_id not in AGY_ORCHESTRATOR_SPECS or not requires_agy_cli_orchestrator_session(raw):
        variants = getattr(root_opt, "selection_variants", ()) or ()
        if variants:
            return any(getattr(v, "name", None) == raw for v in variants)
        return raw == parsed.model_id

    spec = AGY_ORCHESTRATOR_SPECS[parsed.model_id]
    planner_model = parsed.planner_model or spec.planner_model_id
    executor_model = parsed.executor_model or spec.executor_model_id
    if (
        not is_valid_agy_planner_model(planner_model)
        or not is_valid_agy_executor_model(executor_model)
    ):
        return False
    planner_opt = by_name.get(planner_model)
    executor_opt = by_name.get(executor_model)
    if planner_opt is None or executor_opt is None:
        variants = getattr(root_opt, "selection_variants", ()) or ()
        return any(getattr(v, "name", None) == raw for v in variants)

    for role_opt, role_effort in (
        (planner_opt, parsed.planner_effort),
        (executor_opt, parsed.executor_effort),
    ):
        if is_agy_orchestrator_model(getattr(role_opt, "name", "")):
            return False
        efforts = tuple(getattr(role_opt, "reasoning_efforts", ()) or ())
        if not efforts:
            efforts = tuple(
                dict.fromkeys(
                    str(getattr(v, "effort", "") or "").strip().lower()
                    for v in (getattr(role_opt, "selection_variants", ()) or ())
                    if str(getattr(v, "effort", "") or "").strip()
                )
            )
        if efforts:
            if not role_effort or role_effort not in efforts:
                return False
        elif role_effort is not None:
            return False
    return True


def normalize_agy_model_input(
    raw: str | None,
    known_model_names: Sequence[str] = (),
) -> str:
    """Normalize AGY CLI hyphenated slugs or Orchestrator shorthands."""
    text = str(raw or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered == "orchestrator":
        text = AGY_ORCHESTRATOR_MODE_ID
    elif lowered.startswith("orchestrator/"):
        text = f"{AGY_ORCHESTRATOR_MODE_ID}/{text[len('orchestrator/'):]}"
    known_list = [str(name or "").strip() for name in (known_model_names or ()) if str(name or "").strip()]
    known_set = set(known_list)
    if text in known_set:
        return text

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
                if not known_list or (
                    base in known_set and not any(c.startswith(f"{base}/") for c in known_list)
                ):
                    return same_target
        elif len(parts) in {3, 5} and parts[0].strip() in AGY_ORCHESTRATOR_SPECS:
            try:
                parsed = parse_agy_model_selection(text)
                return compose_agy_orchestrator_selection(
                    parsed.model_id,
                    parsed.planner_effort,
                    parsed.executor_effort,
                    planner_model=parsed.planner_model,
                    executor_model=parsed.executor_model,
                )
            except ValueError:
                return text
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
