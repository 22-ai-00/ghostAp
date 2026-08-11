"""Pure model capability resolution for explicit ACP configuration cards."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.acp.options import ACPModelOption, ACPModelSelectionVariant


@dataclass(frozen=True)
class ModelSelection:
    model: str
    profile: str | None
    effort: str | None
    value: str


@dataclass(frozen=True)
class ModelCascadeState:
    model_names: tuple[str, ...]
    selected_model: str
    profiles: tuple[str, ...]
    selected_profile: str | None
    efforts: tuple[str, ...]
    selected_effort: str | None
    selection: str
    selections: tuple[ModelSelection, ...] = ()


def _ordered_unique(values: Sequence[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _variants(models: Sequence[ACPModelOption]) -> tuple[ACPModelSelectionVariant, ...]:
    variants: list[ACPModelSelectionVariant] = []
    seen: set[str] = set()
    for option in models:
        declared = option.selection_variants or (
            ACPModelSelectionVariant(
                name=option.name,
                model=option.name,
                is_default=option.is_default,
            ),
        )
        for variant in declared:
            if variant.name and variant.model and variant.name not in seen:
                seen.add(variant.name)
                variants.append(variant)
    return tuple(variants)


def available_model_names(models: Sequence[ACPModelOption]) -> tuple[str, ...]:
    return _ordered_unique(tuple(variant.model for variant in _variants(models)))


def available_model_profiles(
    models: Sequence[ACPModelOption], model: str
) -> tuple[str, ...]:
    return _ordered_unique(
        tuple(
            variant.profile
            for variant in _variants(models)
            if variant.model == model
        )
    )


def available_model_efforts(
    models: Sequence[ACPModelOption],
    model: str,
    profile: str | None = None,
) -> tuple[str, ...]:
    return _ordered_unique(
        tuple(
            variant.effort
            for variant in _variants(models)
            if variant.model == model and variant.profile == profile
        )
    )


def parse_model_selection(
    models: Sequence[ACPModelOption], selection: str | None
) -> ModelSelection | None:
    value = str(selection or "").strip()
    variant = next(
        (candidate for candidate in _variants(models) if candidate.name == value),
        None,
    )
    if variant is None:
        return None
    return ModelSelection(
        model=variant.model,
        profile=variant.profile,
        effort=variant.effort,
        value=variant.name,
    )


def available_model_selections(
    models: Sequence[ACPModelOption],
) -> tuple[ModelSelection, ...]:
    """Return every backend-declared exact selection in stable display order."""
    return tuple(
        ModelSelection(
            model=variant.model,
            profile=variant.profile,
            effort=variant.effort,
            value=variant.name,
        )
        for variant in _variants(models)
    )


def compose_model_selection(
    models: Sequence[ACPModelOption],
    *,
    model: str,
    profile: str | None = None,
    effort: str | None = None,
) -> str | None:
    normalized_profile = str(profile or "").strip().lower() or None
    normalized_effort = str(effort or "").strip().lower() or None
    variant = next(
        (
            candidate
            for candidate in _variants(models)
            if candidate.model == str(model or "").strip()
            and candidate.profile == normalized_profile
            and candidate.effort == normalized_effort
        ),
        None,
    )
    return variant.name if variant is not None else None


def validate_model_selection(
    models: Sequence[ACPModelOption], selection: str | None
) -> bool:
    return parse_model_selection(models, selection) is not None


def _model_default(
    models: Sequence[ACPModelOption], names: tuple[str, ...]
) -> str:
    return next(
        (option.name for option in models if option.is_default and option.name in names),
        names[0],
    )


def _local_default_effort(
    models: Sequence[ACPModelOption],
    model: str,
    profile: str | None,
    efforts: tuple[str, ...],
) -> str | None:
    variants = _variants(models)
    declared = next(
        (
            variant.effort
            for variant in variants
            if variant.model == model
            and variant.profile == profile
            and variant.effort in efforts
            and variant.is_default
        ),
        None,
    )
    if declared is not None:
        return declared
    option_default = next(
        (
            option.default_reasoning_effort
            for option in models
            if option.name == model
            and option.default_reasoning_effort in efforts
        ),
        None,
    )
    return option_default or (efforts[0] if efforts else None)


def resolve_model_cascade(
    models: Sequence[ACPModelOption],
    *,
    current_model: str | None = None,
    selected_model: str | None = None,
    selected_profile: str | None = None,
    selected_effort: str | None = None,
) -> ModelCascadeState:
    names = available_model_names(models)
    if not names:
        return ModelCascadeState((), "", (), None, (), None, "", ())

    current = parse_model_selection(models, current_model)
    pending_model = str(selected_model or "").strip()
    model = (
        pending_model
        if pending_model in names
        else current.model
        if current is not None
        else _model_default(models, names)
    )
    profiles = available_model_profiles(models, model)
    pending_profile = str(selected_profile or "").strip().lower() or None
    profile = None
    if profiles:
        if pending_profile in profiles:
            profile = pending_profile
        elif current is not None and current.model == model and current.profile in profiles:
            profile = current.profile
        else:
            profile = "standard" if "standard" in profiles else profiles[0]

    efforts = available_model_efforts(models, model, profile)
    pending_effort = str(selected_effort or "").strip().lower() or None
    if pending_effort in efforts:
        effort = pending_effort
    elif (
        current is not None
        and current.model == model
        and current.profile == profile
        and current.effort in efforts
    ):
        effort = current.effort
    else:
        effort = _local_default_effort(models, model, profile, efforts)

    selection = compose_model_selection(
        models,
        model=model,
        profile=profile,
        effort=effort,
    )
    if selection is None:
        fallback = next(
            (
                variant
                for variant in _variants(models)
                if variant.model == model and variant.profile == profile
            ),
            next(variant for variant in _variants(models) if variant.model == model),
        )
        profile = fallback.profile
        effort = fallback.effort
        selection = fallback.name
        profiles = available_model_profiles(models, model)
        efforts = available_model_efforts(models, model, profile)

    return ModelCascadeState(
        model_names=names,
        selected_model=model,
        profiles=profiles,
        selected_profile=profile,
        efforts=efforts,
        selected_effort=effort,
        selection=selection,
        selections=available_model_selections(models),
    )
