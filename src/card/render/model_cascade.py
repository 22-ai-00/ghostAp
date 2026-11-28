"""Pure model capability resolution for explicit ACP configuration cards."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.acp.agy_selection import (
    AGY_DEFAULT_MODEL_ID,
    AGY_ORCHESTRATOR_DEFAULT_PROFILE,
    compose_agy_orchestrator_selection,
    filter_agy_executor_models,
    filter_agy_planner_models,
    get_agy_orchestrator_spec,
    is_agy_orchestrator_model,
    is_valid_agy_catalog_selection,
    parse_agy_model_selection,
    requires_agy_cli_orchestrator_session,
    resolve_native_orchestrator_preset,
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
    planner_model_names: tuple[str, ...] = ()
    executor_model_names: tuple[str, ...] = ()


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


def _role_model_efforts(
    by_name: dict[str, ACPModelOption],
    models: Sequence[ACPModelOption],
    role_model: str,
) -> tuple[str, ...]:
    opt = by_name[role_model]
    efforts = tuple(
        dict.fromkeys(
            str(eff).strip().lower()
            for eff in (opt.reasoning_efforts or ())
            if str(eff or "").strip()
        )
    )
    if not efforts:
        efforts = available_model_efforts(models, role_model, None)
    return efforts


def _resolve_custom_role_effort(
    raw_efforts: tuple[str, ...],
    *,
    pending_effort: str | None,
    cur_effort: str | None,
    same_role_model: bool,
) -> str:
    eff = (
        None
        if pending_effort in {None, AGY_ORCHESTRATOR_DEFAULT_PROFILE}
        else pending_effort
    )
    if eff in raw_efforts:
        return eff
    if same_role_model and cur_effort in raw_efforts:
        return cur_effort
    if "high" in raw_efforts:
        return "high"
    return raw_efforts[0]


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
    all_catalog_names = tuple(
        dict.fromkeys(
            str(getattr(opt, "name", "") or "").strip()
            for opt in (models or ())
            if str(getattr(opt, "name", "") or "").strip()
        )
    )
    role_options = [
        opt for opt in (models or ())
        if getattr(opt, "name", None) and not is_agy_orchestrator_model(opt.name)
    ]
    by_name = {opt.name: opt for opt in role_options}
    planner_model_names = filter_agy_planner_models(tuple(by_name.keys())) if by_name else ()
    executor_model_names = filter_agy_executor_models(tuple(by_name.keys())) if by_name else ()
    role_model_names = _ordered_unique((*planner_model_names, *executor_model_names))
    has_role_models = bool(
        spec is not None
        and spec.planner_model_id in planner_model_names
        and spec.executor_model_id in executor_model_names
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
            planner_model_names=planner_model_names,
            executor_model_names=executor_model_names,
        )

    parsed_cur = None
    cur_spec = None
    cur_p_model = None
    cur_e_model = None
    cur_is_bare = False
    if current_model and is_agy_orchestrator_model(current_model):
        try:
            cand = parse_agy_model_selection(current_model)
            cur_spec = get_agy_orchestrator_spec(cand.model_id)
            if cur_spec is not None:
                parsed_cur = cand
                cur_p_model = cand.planner_model or cur_spec.planner_model_id
                cur_e_model = cand.executor_model or cur_spec.executor_model_id
                cur_is_bare = not requires_agy_cli_orchestrator_session(current_model)
        except ValueError:
            parsed_cur = None

    pend_p_model = str(pending_planner_model or "").strip()
    if pend_p_model in planner_model_names:
        p_model = pend_p_model
    elif cur_p_model and cur_p_model in planner_model_names:
        p_model = cur_p_model
    elif spec.planner_model_id in planner_model_names:
        p_model = spec.planner_model_id
    else:
        p_model = planner_model_names[0]

    pend_e_model = str(pending_executor_model or "").strip()
    if pend_e_model in executor_model_names:
        e_model = pend_e_model
    elif cur_e_model and cur_e_model in executor_model_names:
        e_model = cur_e_model
    elif spec.executor_model_id in executor_model_names:
        e_model = spec.executor_model_id
    else:
        e_model = executor_model_names[0]

    raw_p_efforts = _role_model_efforts(by_name, models, p_model)
    raw_e_efforts = _role_model_efforts(by_name, models, e_model)
    native_preset = (
        resolve_native_orchestrator_preset(p_model, all_catalog_names)
        if e_model == "gemini-4-hydrogen"
        else None
    )
    pend_p_eff = str(pending_planner_effort or "").strip().lower() or None
    pend_e_eff = str(pending_executor_effort or "").strip().lower() or None
    native_default_active = bool(
        native_preset is not None
        and (
            (
                pend_p_eff == AGY_ORCHESTRATOR_DEFAULT_PROFILE
                and pend_e_eff in {None, AGY_ORCHESTRATOR_DEFAULT_PROFILE}
            )
            or (
                pend_p_eff is None
                and pend_e_eff in {None, AGY_ORCHESTRATOR_DEFAULT_PROFILE}
                and cur_is_bare
                and pending_planner_model is None
                and pending_executor_model is None
            )
        )
    )

    if native_default_active:
        assert native_preset is not None
        p_efforts: tuple[str, ...] = (AGY_ORCHESTRATOR_DEFAULT_PROFILE, *raw_p_efforts)
        e_efforts: tuple[str, ...] = (AGY_ORCHESTRATOR_DEFAULT_PROFILE, *raw_e_efforts)
        p_eff: str | None = AGY_ORCHESTRATOR_DEFAULT_PROFILE
        e_eff: str | None = AGY_ORCHESTRATOR_DEFAULT_PROFILE
        selection = native_preset
    elif not raw_p_efforts or not raw_e_efforts:
        if native_preset is not None:
            p_efforts = (AGY_ORCHESTRATOR_DEFAULT_PROFILE, *raw_p_efforts)
            e_efforts = (AGY_ORCHESTRATOR_DEFAULT_PROFILE, *raw_e_efforts)
            p_eff = AGY_ORCHESTRATOR_DEFAULT_PROFILE
            e_eff = AGY_ORCHESTRATOR_DEFAULT_PROFILE
            selection = native_preset
        else:
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
                planner_model_names=planner_model_names,
                executor_model_names=executor_model_names,
            )
    else:
        p_efforts = (
            (AGY_ORCHESTRATOR_DEFAULT_PROFILE, *raw_p_efforts)
            if native_preset is not None
            else raw_p_efforts
        )
        e_efforts = raw_e_efforts
        p_eff = _resolve_custom_role_effort(
            raw_p_efforts,
            pending_effort=pend_p_eff,
            cur_effort=parsed_cur.planner_effort if parsed_cur else None,
            same_role_model=(cur_p_model == p_model),
        )
        e_eff = _resolve_custom_role_effort(
            raw_e_efforts,
            pending_effort=pend_e_eff,
            cur_effort=parsed_cur.executor_effort if parsed_cur else None,
            same_role_model=(cur_e_model == e_model),
        )
        target_orch = native_preset or (
            AGY_DEFAULT_MODEL_ID if AGY_DEFAULT_MODEL_ID in all_catalog_names else orch_id
        )
        selection = compose_agy_orchestrator_selection(
            target_orch,
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
        planner_model_names=planner_model_names,
        executor_model_names=executor_model_names,
    )
