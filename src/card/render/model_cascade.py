"""Pure model capability resolution for explicit ACP configuration cards."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.acp.agy_selection import (
    AGY_DEFAULT_MODEL_ID,
    AGY_ORCHESTRATOR_DEFAULT_PROFILE,
    compose_agy_orchestrator_selection,
    get_agy_orchestrator_spec,
    is_agy_orchestrator_model,
    is_valid_agy_catalog_selection,
    parse_agy_model_selection,
)
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


@dataclass(frozen=True)
class AGYOrchestratorCascadeState:
    has_role_models: bool
    orchestrator_model: str
    role_model_names: tuple[str, ...]
    selected_planner_model: str
    planner_efforts: tuple[str, ...]
    selected_planner_effort: str | None
    selected_executor_model: str
    executor_efforts: tuple[str, ...]
    selected_executor_effort: str | None
    selection: str


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
    if variant is not None:
        return ModelSelection(
            model=variant.model,
            profile=variant.profile,
            effort=variant.effort,
            value=variant.name,
        )
    if value and is_valid_agy_catalog_selection(models, value):
        parsed = parse_agy_model_selection(value)
        return ModelSelection(
            model=parsed.model_id,
            profile=parsed.planner_effort,
            effort=parsed.executor_effort if parsed.planner_model or parsed.executor_model else parsed.thought_level,
            value=value,
        )
    return None


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


def _resolve_agy_role(
    by_name: dict[str, ACPModelOption],
    models: Sequence[ACPModelOption],
    *,
    pending_model: str | None,
    pending_effort: str | None,
    cur_model: str | None,
    cur_effort: str | None,
    default_role_model_id: str,
) -> tuple[str, tuple[str, ...], str | None]:
    pend_m = str(pending_model or "").strip()
    if pend_m in by_name:
        role_model = pend_m
    elif cur_model and cur_model in by_name:
        role_model = cur_model
    else:
        role_model = default_role_model_id

    opt = by_name[role_model]
    efforts = tuple(dict.fromkeys(eff for eff in (opt.reasoning_efforts or ()) if eff))
    if not efforts:
        efforts = available_model_efforts(models, role_model, None)
    if not efforts:
        return role_model, (), None

    pend_e = str(pending_effort or "").strip().lower() or None
    if pend_e == AGY_ORCHESTRATOR_DEFAULT_PROFILE:
        pend_e = None
    if pend_e in efforts:
        role_effort: str | None = pend_e
    elif (cur_model is None or cur_model == role_model) and cur_effort in efforts:
        role_effort = cur_effort
    elif "high" in efforts:
        role_effort = "high"
    elif opt.default_reasoning_effort in efforts:
        role_effort = opt.default_reasoning_effort
    else:
        role_effort = efforts[0]
    return role_model, efforts, role_effort


def resolve_agy_orchestrator_cascade(
    models: Sequence[ACPModelOption],
    *,
    orchestrator_model: str | None = None,
    current_model: str | None = None,
    pending_planner_model: str | None = None,
    pending_planner_effort: str | None = None,
    pending_executor_model: str | None = None,
    pending_executor_effort: str | None = None,
) -> AGYOrchestratorCascadeState:
    """Resolve 4-axis Planner/Executor model + effort state for AGY Orchestrators."""
    orch_id = str(orchestrator_model or AGY_DEFAULT_MODEL_ID).strip()
    spec = get_agy_orchestrator_spec(orch_id)
    role_options = [
        opt for opt in (models or ())
        if getattr(opt, "name", None) and not is_agy_orchestrator_model(opt.name)
    ]
    role_model_names = tuple(dict.fromkeys(opt.name for opt in role_options))
    by_name = {opt.name: opt for opt in role_options}
    has_role_models = bool(
        spec is not None
        and spec.planner_model_id in by_name
        and spec.executor_model_id in by_name
    )
    if not has_role_models or spec is None:
        return AGYOrchestratorCascadeState(
            has_role_models=False,
            orchestrator_model=orch_id,
            role_model_names=role_model_names,
            selected_planner_model="",
            planner_efforts=(),
            selected_planner_effort=None,
            selected_executor_model="",
            executor_efforts=(),
            selected_executor_effort=None,
            selection=orch_id,
        )

    parsed_cur = None
    if current_model:
        try:
            cand = parse_agy_model_selection(current_model)
            if cand.model_id == orch_id:
                parsed_cur = cand
        except ValueError:
            parsed_cur = None

    p_model, p_efforts, p_eff = _resolve_agy_role(
        by_name,
        models,
        pending_model=pending_planner_model,
        pending_effort=pending_planner_effort,
        cur_model=parsed_cur.planner_model if parsed_cur else None,
        cur_effort=parsed_cur.planner_effort if parsed_cur else None,
        default_role_model_id=spec.planner_model_id,
    )
    e_model, e_efforts, e_eff = _resolve_agy_role(
        by_name,
        models,
        pending_model=pending_executor_model,
        pending_effort=pending_executor_effort,
        cur_model=parsed_cur.executor_model if parsed_cur else None,
        cur_effort=parsed_cur.executor_effort if parsed_cur else None,
        default_role_model_id=spec.executor_model_id,
    )
    selection = compose_agy_orchestrator_selection(
        orch_id,
        p_eff,
        e_eff,
        planner_model=p_model,
        executor_model=e_model,
    )
    return AGYOrchestratorCascadeState(
        has_role_models=True,
        orchestrator_model=orch_id,
        role_model_names=role_model_names,
        selected_planner_model=p_model,
        planner_efforts=p_efforts,
        selected_planner_effort=p_eff,
        selected_executor_model=e_model,
        executor_efforts=e_efforts,
        selected_executor_effort=e_eff,
        selection=selection,
    )
