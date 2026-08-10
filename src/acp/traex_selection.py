"""Traex model/Profile/Effort selection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .model_selection import CODEX_REASONING_EFFORTS

_PROFILES = frozenset({"standard", "max"})


@dataclass(frozen=True)
class TraexProfileMetadata:
    profile: str
    backend_model_value: str
    reasoning_efforts: tuple[str, ...] = ()
    default_effort: Optional[str] = None


@dataclass(frozen=True)
class TraexModelMetadata:
    config_name: str
    slug: str
    profiles: tuple[TraexProfileMetadata, ...]


@dataclass(frozen=True)
class TraexRuntimeSelection:
    model_id: str
    backend_model_value: str
    profile: str
    effort: Optional[str]


def compose_traex_model_selection(
    model_id: str,
    profile: str,
    effort: Optional[str],
) -> str:
    model = str(model_id or "").strip()
    selected_profile = str(profile or "standard").strip().lower()
    selected_effort = str(effort or "").strip().lower()
    if not model:
        return ""
    if selected_effort:
        return f"{model}/{selected_profile}/{selected_effort}"
    if selected_profile != "standard":
        return f"{model}/{selected_profile}"
    return model


def split_traex_model_selection(
    value: Optional[str],
) -> tuple[str, str, Optional[str]]:
    selection = str(value or "").strip()
    if not selection:
        return "", "standard", None
    parts = [part for part in selection.split("/") if part]
    if len(parts) >= 3 and parts[-1].lower() in CODEX_REASONING_EFFORTS:
        return "/".join(parts[:-2]), parts[-2].lower(), parts[-1].lower()
    if len(parts) >= 2 and parts[-1].lower() in _PROFILES:
        return "/".join(parts[:-1]), parts[-1].lower(), None
    if len(parts) >= 2 and parts[-1].lower() in CODEX_REASONING_EFFORTS:
        suffix = parts[-1].lower()
        if suffix == "max":
            return "/".join(parts[:-1]), "max", None
        return "/".join(parts[:-1]), "standard", suffix
    return selection, "standard", None


def _reasoning_efforts(raw: object) -> tuple[str, ...]:
    values: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        effort = str(item.get("effort") or "").strip().lower()
        if effort in CODEX_REASONING_EFFORTS and effort not in values:
            values.append(effort)
    return tuple(values)


def _default_effort(value: object, supported: tuple[str, ...]) -> Optional[str]:
    effort = str(value or "").strip().lower()
    return effort if effort in supported else (supported[0] if supported else None)


def load_traex_model_metadata(
    metadata_path: Optional[Path] = None,
) -> tuple[TraexModelMetadata, ...]:
    path = metadata_path or (Path.home() / ".trae" / "cli" / "models_cache.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ()
    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        return ()

    models: list[TraexModelMetadata] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        slug = str(raw.get("slug") or "").strip()
        config_name = str(raw.get("config_name") or slug).strip()
        if not config_name:
            continue
        business = raw.get("business_metadata")
        variants = business.get("variants") if isinstance(business, dict) else None
        variants = variants if isinstance(variants, dict) else {}

        standard_efforts = _reasoning_efforts(
            variants.get("standard_supported_reasoning_levels")
            or raw.get("supported_reasoning_levels")
        )
        standard_default = _default_effort(
            variants.get("standard_default_reasoning_level")
            or raw.get("default_reasoning_level"),
            standard_efforts,
        )
        profiles = [
            TraexProfileMetadata(
                profile="standard",
                backend_model_value=config_name,
                reasoning_efforts=standard_efforts,
                default_effort=standard_default,
            )
        ]

        max_key = str(variants.get("max_key") or "").strip()
        if max_key:
            max_efforts = _reasoning_efforts(
                variants.get("max_supported_reasoning_levels")
            )
            profiles.append(
                TraexProfileMetadata(
                    profile="max",
                    backend_model_value=max_key,
                    reasoning_efforts=max_efforts,
                    default_effort=_default_effort(
                        variants.get("max_default_reasoning_level"),
                        max_efforts,
                    ),
                )
            )

        models.append(
            TraexModelMetadata(
                config_name=config_name,
                slug=slug or config_name,
                profiles=tuple(profiles),
            )
        )
    return tuple(models)


def find_traex_model_metadata(
    model_id: str,
    *,
    metadata_path: Optional[Path] = None,
) -> Optional[TraexModelMetadata]:
    target = str(model_id or "").strip()
    for model in load_traex_model_metadata(metadata_path):
        if target in {model.config_name, model.slug}:
            return model
    return None


def resolve_traex_runtime_selection(
    value: Optional[str],
    *,
    metadata_path: Optional[Path] = None,
) -> TraexRuntimeSelection:
    raw_model, profile, effort = split_traex_model_selection(value)
    if not raw_model:
        raise ValueError("Traex model is empty")
    if profile not in _PROFILES:
        raise ValueError(f"Unsupported Traex profile: {profile}")

    metadata = find_traex_model_metadata(raw_model, metadata_path=metadata_path)
    if metadata is None:
        if profile != "standard":
            raise ValueError(f"Unsupported Traex profile without metadata: {profile}")
        if effort is not None and effort not in CODEX_REASONING_EFFORTS:
            raise ValueError(f"Unsupported Traex effort: {effort}")
        return TraexRuntimeSelection(
            model_id=raw_model,
            backend_model_value=raw_model,
            profile="standard",
            effort=effort,
        )

    selected_profile = next(
        (item for item in metadata.profiles if item.profile == profile),
        None,
    )
    if selected_profile is None:
        raise ValueError(f"Unsupported Traex profile: {profile}")
    if effort is not None and effort not in selected_profile.reasoning_efforts:
        raise ValueError(f"Unsupported Traex effort for {profile}: {effort}")
    return TraexRuntimeSelection(
        model_id=metadata.config_name,
        backend_model_value=selected_profile.backend_model_value,
        profile=profile,
        effort=effort,
    )
