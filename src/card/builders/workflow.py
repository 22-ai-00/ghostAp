"""Pure Card JSON 2.0 builders for Workflow (/wf, /wf_config, /wf_help) surfaces.

Strict import boundary: this module MUST NEVER import ``src.card.delivery``,
``src.card.session``, or ``src.feishu``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.card.actions import dispatch as action_ids
from src.card.render.model_cascade import resolve_model_cascade
from src.card.schema import require_card_json_2
from src.card.shared import build_callback_button, build_responsive_layout
from src.card.shared.text_safety import neutralize_feishu_rich_text_controls
from src.card.shared.truncation import count_tagged_nodes
from src.card.thresholds import THRESHOLDS
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.constants import MAX_WORKFLOW_AGENT_POOL_SIZE
from src.workflow_engine.tool_registry import ToolCapability

_MAX_SELECT_OPTIONS = 25
_CARD_BYTE_BUDGET = THRESHOLDS["CARD_BYTE_BUDGET"]
_CARD_NODE_BUDGET = THRESHOLDS["CARD_NODE_BUDGET"]

WORKFLOW_PRESET_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("collaborative", "推荐协作组", "Planner + Workers + Verifier + Synthesizer 全闭环"),
    ("quick", "快速双人", "Worker + Verifier 轻量快速交付"),
    ("review_team", "深度复核", "多工具独立实现 + 交叉评审复核"),
    ("deep_collab", "深度协作", "Scout + Planner + Workers + 对抗复核 + 汇总"),
    ("auto", "自动推荐", "根据当前可用工具自动分配最佳协作团队"),
    ("custom", "自定义", "按成员逐个定制 tool / model / effort / role"),
)

WORKFLOW_ROLE_CATALOG: tuple[tuple[str, str], ...] = (
    ("scout", "scout · 侦察/调研"),
    ("planner", "planner · 规划/拆解"),
    ("worker", "worker · 实现/执行"),
    ("verifier", "verifier · 校验/复核"),
    ("synthesizer", "synthesizer · 汇总/收口"),
)

_DEFAULT_EFFORTS_CLAUDE: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
_DEFAULT_EFFORTS_STANDARD: tuple[str, ...] = ("low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class WorkflowConfigEditorState:
    """Immutable view-model passed into ``build_workflow_config_card``."""

    config_session_id: str = ""
    chat_id: str = ""
    project_id: str | None = None
    thread_root_id: str | None = None
    preset_id: str = "collaborative"
    members: tuple[WorkflowAgentBinding, ...] = ()
    orchestrator_agent_id: str = ""
    selected_agent_id: str = ""
    editor_role: str = "worker"
    editor_tool: str = ""
    editor_model: str | None = None
    editor_profile: str | None = None
    editor_effort: str | None = None
    requirement: str | None = None
    saved_as_default: bool = False
    status_notice: str | None = None


def _strip_surrogates(value: Any) -> str:
    return str(value or "").encode("utf-8", errors="replace").decode("utf-8")


def _safe_one_line(value: Any, *, limit: int = 120) -> str:
    text = neutralize_feishu_rich_text_controls(_strip_surrogates(value))
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _escape_md(value: Any, *, limit: int = 160) -> str:
    text = _safe_one_line(value, limit=limit)
    for ch in ("*", "_", "`", "|", "[", "]", "~"):
        text = text.replace(ch, "\\" + ch)
    return text


def _fit_summary(text: str, *, default: str) -> str:
    safe_default = _strip_surrogates(default)
    clean = " ".join(_strip_surrogates(text).split()) or safe_default
    if len(clean) < 8:
        clean = f"Workflow · {clean}"
    if len(clean) < 8:
        clean = safe_default
    if len(clean) > 60:
        clean = clean[:59].rstrip() + "…"
    return clean


def _default_efforts_for_tool(tool_name: str) -> tuple[str, ...]:
    tool = str(tool_name or "").strip().lower()
    if tool in {"claude", "claude_w"}:
        return _DEFAULT_EFFORTS_CLAUDE
    return _DEFAULT_EFFORTS_STANDARD


def _cap_options_preserving(
    items: Sequence[str],
    *,
    keep: Sequence[str | None] = (),
    limit: int = _MAX_SELECT_OPTIONS,
) -> tuple[str, ...]:
    ordered = list(dict.fromkeys(str(item) for item in items if str(item or "").strip()))
    if limit <= 0:
        return ()
    if len(ordered) <= limit:
        return tuple(ordered)
    result = ordered[:limit]
    result_set = set(result)
    ordered_set = set(ordered)
    keep_in_ordered: list[str] = []
    for target in keep:
        if not target:
            continue
        target_str = str(target)
        if target_str in ordered_set and target_str not in keep_in_ordered:
            keep_in_ordered.append(target_str)
    if not keep_in_ordered or all(item in result_set for item in keep_in_ordered):
        return tuple(result)
    keep_in_ordered = keep_in_ordered[:limit]
    keep_set = set(keep_in_ordered)
    kept_head = [item for item in result if item not in keep_set][
        : max(0, limit - len(keep_in_ordered))
    ]
    return tuple([*kept_head, *keep_in_ordered])


def _extract_current_base_model(
    cap: ToolCapability,
    tool: str,
    raw_current: str | None,
) -> str | None:
    if not raw_current:
        return None
    if raw_current.startswith("["):
        if tool == "dsh":
            from src.acp.dsh_selection import decode_dsh_model_value, split_dsh_model_selection

            try:
                model_json, _reasoning = split_dsh_model_selection(raw_current)
                provider, model_id = decode_dsh_model_value(model_json)
                return f"{provider}/{model_id}"
            except ValueError:
                return None
        return None
    if not (
        cap.supports_effort
        and tool in {"traex", "codex", "codex_w", "claude", "claude_w", "agy", "agy_b"}
    ):
        return raw_current

    if cap.models:
        from src.acp.model_selection import CODEX_REASONING_EFFORTS

        matched_bases: list[str] = []
        for candidate in sorted(cap.models, key=len, reverse=True):
            prefix = f"{candidate}/"
            if not raw_current.startswith(prefix):
                continue
            remainder = raw_current[len(prefix) :].strip().lower()
            if not remainder:
                continue
            rem_parts = [part for part in remainder.split("/") if part]
            allowed_efforts = set(cap.efforts_by_model.get(candidate, ())) or set(
                CODEX_REASONING_EFFORTS
            )
            allowed_profiles = set(cap.profiles_by_model.get(candidate, ()))
            if tool == "traex":
                allowed_profiles |= {"standard", "max"}
            if (
                len(rem_parts) == 1
                and (rem_parts[0] in allowed_efforts or rem_parts[0] in allowed_profiles)
            ) or (
                len(rem_parts) == 2
                and (
                    (rem_parts[0] in allowed_profiles and rem_parts[1] in allowed_efforts)
                    or (
                        tool in {"agy", "agy_b"}
                        and rem_parts[0] in allowed_efforts
                        and rem_parts[1] in allowed_efforts
                    )
                )
            ):
                matched_bases.append(candidate)
        if matched_bases:
            return matched_bases[0]
        if raw_current in cap.models:
            return raw_current

    if tool == "traex":
        from src.acp.traex_selection import split_traex_model_selection

        base, _profile, _effort = split_traex_model_selection(raw_current)
        return base or None
    if tool in {"codex", "codex_w"}:
        from src.acp.model_selection import split_codex_model_selection

        base, _effort = split_codex_model_selection(raw_current)
        return base or None
    if tool in {"claude", "claude_w"}:
        from src.acp.claude_selection import split_claude_model_selection

        base, _effort = split_claude_model_selection(raw_current)
        return base or None
    if tool in {"agy", "agy_b"}:
        from src.acp.agy_selection import (
            AGY_REASONING_EFFORTS,
            is_agy_orchestrator_model,
            parse_agy_model_selection,
        )
        from src.acp.model_selection import split_slash_model_effort

        if is_agy_orchestrator_model(raw_current):
            try:
                return parse_agy_model_selection(raw_current).model_id
            except ValueError:
                return raw_current.split("/", 1)[0].strip() or None
        base, _effort = split_slash_model_effort(
            raw_current,
            allowed_efforts=AGY_REASONING_EFFORTS,
        )
        return base or None
    return raw_current


def capability_to_acp_options(
    cap: ToolCapability,
    *,
    current_model: str | None = None,
) -> list[ACPModelOption]:
    """Convert a ``ToolCapability`` snapshot into ``ACPModelOption`` entries.

    Provides a cold-cache ``"default"`` option with tool-appropriate reasoning
    efforts when ``cap.supports_effort`` is true and ``cap.models`` is empty.
    """
    tool = str(cap.tool_name or "").strip().lower()
    raw_current = str(current_model or "").strip() or None
    current_base = _extract_current_base_model(cap, tool, raw_current)

    if not cap.models:
        if not cap.supports_effort:
            if current_base:
                return [
                    ACPModelOption(
                        name=current_base,
                        is_default=True,
                        selection_variants=(
                            ACPModelSelectionVariant(
                                name=current_base,
                                model=current_base,
                                is_default=True,
                            ),
                        ),
                    )
                ]
            return []

        efforts = _default_efforts_for_tool(tool)
        default_effort = (
            cap.default_effort if cap.default_effort in efforts else efforts[0]
        )
        candidate_models = ["default"]
        if current_base and current_base != "default":
            candidate_models.append(current_base)

        options: list[ACPModelOption] = []
        for idx, model_name in enumerate(candidate_models):
            variants = tuple(
                ACPModelSelectionVariant(
                    name=f"{model_name}/{effort}",
                    model=model_name,
                    effort=effort,
                    is_default=(idx == 0 and effort == default_effort),
                )
                for effort in efforts
            )
            options.append(
                ACPModelOption(
                    name=model_name,
                    is_default=(idx == 0),
                    selection_variants=variants,
                    reasoning_efforts=efforts,
                    default_reasoning_effort=default_effort,
                )
            )
        return options

    selected_models = _cap_options_preserving(
        cap.models,
        keep=(current_base, raw_current),
        limit=_MAX_SELECT_OPTIONS,
    )
    options = []
    for idx, model_name in enumerate(selected_models):
        is_default_model = idx == 0
        profiles = tuple(cap.profiles_by_model.get(model_name, ()))
        if cap.supports_effort:
            efforts = tuple(cap.efforts_by_model.get(model_name, ()))
            if not efforts and not cap.authoritative_models:
                efforts = _default_efforts_for_tool(tool)
            default_effort = (
                cap.default_effort if cap.default_effort in efforts else (efforts[0] if efforts else None)
            )
        else:
            efforts = ()
            default_effort = None

        variants_list: list[ACPModelSelectionVariant] = []
        if profiles and efforts:
            for profile in profiles:
                for effort in efforts:
                    variants_list.append(
                        ACPModelSelectionVariant(
                            name=f"{model_name}/{profile}/{effort}",
                            model=model_name,
                            profile=profile,
                            effort=effort,
                            is_default=(
                                is_default_model
                                and profile == ("standard" if "standard" in profiles else profiles[0])
                                and effort == default_effort
                            ),
                        )
                    )
        elif efforts:
            for effort in efforts:
                variants_list.append(
                    ACPModelSelectionVariant(
                        name=f"{model_name}/{effort}",
                        model=model_name,
                        effort=effort,
                        is_default=(is_default_model and effort == default_effort),
                    )
                )
        elif profiles:
            for profile in profiles:
                variants_list.append(
                    ACPModelSelectionVariant(
                        name=f"{model_name}/{profile}",
                        model=model_name,
                        profile=profile,
                        is_default=(is_default_model and profile == profiles[0]),
                    )
                )
        else:
            variants_list.append(
                ACPModelSelectionVariant(
                    name=model_name,
                    model=model_name,
                    is_default=is_default_model,
                )
            )

        options.append(
            ACPModelOption(
                name=model_name,
                is_default=is_default_model,
                selection_variants=tuple(variants_list),
                reasoning_efforts=efforts,
                default_reasoning_effort=default_effort,
            )
        )
    return options


def _collect_tool_efforts(
    cap: ToolCapability,
    options: Sequence[ACPModelOption],
) -> tuple[str, ...]:
    if not cap.supports_effort:
        return ()
    efforts: list[str] = []
    for opt in options:
        for eff in getattr(opt, "reasoning_efforts", ()) or ():
            clean = str(eff or "").strip().lower()
            if clean and clean != "default" and clean not in efforts:
                efforts.append(clean)
        for var in getattr(opt, "selection_variants", ()) or ():
            clean = str(getattr(var, "effort", "") or "").strip().lower()
            if clean and clean != "default" and clean not in efforts:
                efforts.append(clean)
    for model_efforts in cap.efforts_by_model.values():
        for eff in model_efforts:
            clean = str(eff or "").strip().lower()
            if clean and clean != "default" and clean not in efforts:
                efforts.append(clean)
    return tuple(efforts) or _default_efforts_for_tool(cap.tool_name)


def _normalize_single_option_variants(
    option: ACPModelOption,
    *,
    supports_effort: bool,
) -> ACPModelOption:
    model_name = option.name
    raw_variants = list(option.selection_variants or ())
    if not raw_variants:
        efforts = (
            tuple(
                dict.fromkeys(
                    str(item).strip().lower()
                    for item in (option.reasoning_efforts or ())
                    if str(item or "").strip()
                )
            )
            if supports_effort
            else ()
        )
        if efforts:
            if "default" not in efforts:
                efforts = ("default", *efforts)
            has_explicit_default = (
                option.default_reasoning_effort in efforts
                and option.default_reasoning_effort != "default"
            )
            synth = tuple(
                ACPModelSelectionVariant(
                    name=f"{model_name}/{eff}" if eff != "default" else f"{model_name}/default",
                    model=model_name,
                    effort=eff,
                    is_default=(
                        eff == option.default_reasoning_effort
                        if has_explicit_default
                        else eff == "default"
                    ),
                )
                for eff in efforts
            )
            return dataclasses.replace(
                option,
                selection_variants=synth,
                reasoning_efforts=efforts,
                default_reasoning_effort=(
                    option.default_reasoning_effort if has_explicit_default else "default"
                ),
            )
        return dataclasses.replace(
            option,
            selection_variants=(
                ACPModelSelectionVariant(
                    name=model_name,
                    model=model_name,
                    is_default=True,
                ),
            ),
            reasoning_efforts=(),
            default_reasoning_effort=None,
        )

    if not supports_effort:
        cleaned_variants: list[ACPModelSelectionVariant] = []
        seen_profiles: set[str | None] = set()
        for variant in raw_variants:
            prof = variant.profile
            if prof in seen_profiles:
                continue
            seen_profiles.add(prof)
            cleaned_variants.append(
                ACPModelSelectionVariant(
                    name=f"{model_name}/{prof}" if prof else model_name,
                    model=model_name,
                    profile=prof,
                    effort=None,
                    is_default=(len(cleaned_variants) == 0),
                )
            )
        return dataclasses.replace(
            option,
            selection_variants=tuple(cleaned_variants),
            reasoning_efforts=(),
            default_reasoning_effort=None,
        )

    by_profile: dict[str | None, list[ACPModelSelectionVariant]] = {}
    for variant in raw_variants:
        by_profile.setdefault(variant.profile, []).append(variant)

    normalized_variants: list[ACPModelSelectionVariant] = []
    seen_names: set[str] = set()
    all_efforts: list[str] = []
    for prof, group in by_profile.items():
        group_efforts = [v.effort for v in group if v.effort]
        if group_efforts and "default" not in group_efforts:
            group_has_default = any(v.is_default and v.effort for v in group) or (
                option.default_reasoning_effort in group_efforts
            )
            default_var_name = (
                f"{model_name}/{prof}/default" if prof else f"{model_name}/default"
            )
            if default_var_name not in seen_names:
                seen_names.add(default_var_name)
                normalized_variants.append(
                    ACPModelSelectionVariant(
                        name=default_var_name,
                        model=model_name,
                        profile=prof,
                        effort="default",
                        is_default=not group_has_default,
                    )
                )
            if "default" not in all_efforts:
                all_efforts.append("default")
        for variant in group:
            if variant.effort and variant.effort not in all_efforts:
                all_efforts.append(variant.effort)
            if variant.name in seen_names:
                continue
            seen_names.add(variant.name)
            normalized_variants.append(variant)

    return dataclasses.replace(
        option,
        selection_variants=tuple(normalized_variants),
        reasoning_efforts=tuple(all_efforts),
    )


def _build_default_workflow_option(
    *,
    existing: ACPModelOption | None,
    tool_efforts: tuple[str, ...],
    supports_effort: bool,
    select_default_model: bool,
) -> ACPModelOption:
    if not supports_effort:
        return ACPModelOption(
            name="default",
            description="default（默认模型）",
            is_default=select_default_model,
            selection_variants=(
                ACPModelSelectionVariant(
                    name="default",
                    model="default",
                    is_default=select_default_model,
                ),
            ),
        )

    efforts_list: list[str] = ["default"]
    if existing is not None:
        for eff in getattr(existing, "reasoning_efforts", ()) or ():
            clean = str(eff or "").strip().lower()
            if clean and clean not in efforts_list:
                efforts_list.append(clean)
        for var in getattr(existing, "selection_variants", ()) or ():
            clean = str(getattr(var, "effort", "") or "").strip().lower()
            if clean and clean not in efforts_list:
                efforts_list.append(clean)
    for eff in tool_efforts:
        if eff and eff not in efforts_list:
            efforts_list.append(eff)
    efforts = tuple(efforts_list)
    variants = tuple(
        ACPModelSelectionVariant(
            name="default" if eff == "default" else f"default/{eff}",
            model="default",
            effort=eff,
            is_default=(eff == "default"),
        )
        for eff in efforts
    )
    return ACPModelOption(
        name="default",
        description="default（默认模型）",
        is_default=select_default_model,
        selection_variants=variants,
        reasoning_efforts=efforts,
        default_reasoning_effort="default",
    )


def normalize_workflow_acp_options(
    cap: ToolCapability,
    raw_options: Sequence[ACPModelOption] | None = None,
    *,
    current_model: str | None = None,
) -> list[ACPModelOption]:
    """Return UI-normalized ACPModelOptions starting with ``default`` for any backend."""
    tool = str(cap.tool_name or "").strip().lower()
    raw_current = str(current_model or "").strip() or None
    current_base = _extract_current_base_model(cap, tool, raw_current)

    source_options = (
        [dataclasses.replace(opt) for opt in raw_options if getattr(opt, "name", None)]
        if raw_options is not None
        else capability_to_acp_options(cap, current_model=current_model)
    )
    by_name: dict[str, ACPModelOption] = {}
    for opt in source_options:
        if opt.name not in by_name:
            by_name[opt.name] = opt

    if (
        current_base
        and current_base != "default"
        and current_base not in by_name
        and not current_base.startswith("[")
    ):
        extra_cap = dataclasses.replace(cap, models=(*cap.models, current_base))
        for synth in capability_to_acp_options(extra_cap, current_model=raw_current):
            if synth.name == current_base:
                by_name[current_base] = synth
                break

    existing_default = by_name.get("default")
    non_default_names = [name for name in by_name if name != "default"]
    capped_non_default = _cap_options_preserving(
        non_default_names,
        keep=(current_base, raw_current),
        limit=max(0, _MAX_SELECT_OPTIONS - 1),
    )

    tool_efforts = _collect_tool_efforts(cap, list(by_name.values()))
    select_default_model = not current_base or current_base == "default"
    normalized: list[ACPModelOption] = [
        _build_default_workflow_option(
            existing=existing_default,
            tool_efforts=tool_efforts,
            supports_effort=cap.supports_effort,
            select_default_model=select_default_model,
        )
    ]
    for name in capped_non_default:
        normalized.append(
            _normalize_single_option_variants(
                by_name[name],
                supports_effort=cap.supports_effort,
            )
        )
    return normalized


def resolve_workflow_cascade_state(
    cap: ToolCapability,
    tool_options: Sequence[ACPModelOption],
    *,
    editor_model: str | None = None,
    editor_profile: str | None = None,
    editor_effort: str | None = None,
) -> tuple[Any, str, str | None, str | None, str | None, str | None, bool]:
    """Resolve cascade state & normalized defaults shared by config card and handler."""
    tool = str(cap.tool_name or "").strip().lower()
    raw_model = str(editor_model or "").strip() or None
    extracted_base = _extract_current_base_model(cap, tool, raw_model)

    selected_model_input = raw_model or "default"
    if (
        raw_model
        and raw_model.startswith("[")
        and tool == "dsh"
        and extracted_base
    ):
        selected_model_input = extracted_base

    embedded_effort: str | None = None
    if editor_effort is None and raw_model:
        for opt in tool_options:
            for var in getattr(opt, "selection_variants", ()) or ():
                if var.name == raw_model and var.effort and var.effort != "default":
                    embedded_effort = var.effort
                    break
            if embedded_effort:
                break
        if not embedded_effort and tool == "dsh" and raw_model.startswith("["):
            from src.acp.dsh_selection import (
                decode_dsh_reasoning_value,
                split_dsh_model_selection,
            )

            try:
                _mj, r_json = split_dsh_model_selection(raw_model)
                decoded = decode_dsh_reasoning_value(r_json) if r_json else None
                if decoded and decoded != "default":
                    embedded_effort = decoded
            except ValueError:
                pass

    selected_effort_input = (
        editor_effort
        if editor_effort is not None
        else (embedded_effort if embedded_effort else "default")
    )
    cascade = resolve_model_cascade(
        tool_options,
        current_model=raw_model,
        selected_model=selected_model_input,
        selected_profile=editor_profile,
        selected_effort=selected_effort_input,
    )
    raw_resolved_model = cascade.selected_model or raw_model or "default"
    resolved_model = None if raw_resolved_model == "default" else raw_resolved_model
    raw_resolved_profile = cascade.selected_profile if cascade.profiles else None
    resolved_profile = (
        None if raw_resolved_profile in (None, "default") else raw_resolved_profile
    )
    raw_resolved_effort = (
        (cascade.selected_effort if cascade.efforts else None)
        if cap.supports_effort
        else None
    )
    has_adjustable_effort = bool(
        cap.supports_effort and any(eff != "default" for eff in cascade.efforts)
    )
    resolved_effort = (
        None
        if (not has_adjustable_effort or raw_resolved_effort in (None, "default"))
        else raw_resolved_effort
    )
    return (
        cascade,
        raw_resolved_model,
        resolved_model,
        raw_resolved_profile,
        resolved_profile,
        resolved_effort,
        has_adjustable_effort,
    )


def _coerce_member(raw: WorkflowAgentBinding | Mapping[str, Any]) -> WorkflowAgentBinding | None:
    if isinstance(raw, WorkflowAgentBinding):
        return raw
    if isinstance(raw, Mapping):
        return WorkflowAgentBinding.from_dict(dict(raw))
    return None


def _coerce_state(state: WorkflowConfigEditorState | Mapping[str, Any]) -> WorkflowConfigEditorState:
    if isinstance(state, WorkflowConfigEditorState):
        return state
    raw_members = state.get("members") or ()
    members = tuple(
        member
        for item in raw_members
        if (member := _coerce_member(item)) is not None
    )
    return WorkflowConfigEditorState(
        config_session_id=str(state.get("config_session_id") or ""),
        chat_id=str(state.get("chat_id") or ""),
        project_id=(str(state.get("project_id")) if state.get("project_id") else None),
        thread_root_id=(str(state.get("thread_root_id")) if state.get("thread_root_id") else None),
        preset_id=str(state.get("preset_id") or "collaborative"),
        members=members,
        orchestrator_agent_id=str(state.get("orchestrator_agent_id") or ""),
        selected_agent_id=str(state.get("selected_agent_id") or ""),
        editor_role=str(state.get("editor_role") or "worker"),
        editor_tool=str(state.get("editor_tool") or ""),
        editor_model=(str(state.get("editor_model")) if state.get("editor_model") else None),
        editor_profile=(str(state.get("editor_profile")) if state.get("editor_profile") else None),
        editor_effort=(str(state.get("editor_effort")) if state.get("editor_effort") else None),
        requirement=(str(state.get("requirement")) if state.get("requirement") else None),
        saved_as_default=bool(state.get("saved_as_default", False)),
        status_notice=(str(state.get("status_notice")) if state.get("status_notice") else None),
    )


def _clean_callback_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, val in raw.items():
        if val is None or val == "":
            continue
        safe_key = _strip_surrogates(key)
        if isinstance(val, str):
            safe_val = _strip_surrogates(val)
            if safe_val:
                cleaned[safe_key] = safe_val
        else:
            cleaned[safe_key] = val
    return cleaned


def _select_static(
    *,
    name: str,
    placeholder: str,
    options: Sequence[tuple[str, str]],
    selected: str | None,
    callback_value: dict[str, Any],
) -> dict[str, Any]:
    cb = _clean_callback_payload(callback_value)
    option_items = [
        {
            "text": {"tag": "plain_text", "content": _safe_one_line(label, limit=64)},
            "value": _strip_surrogates(value),
        }
        for value, label in options
    ]
    element: dict[str, Any] = {
        "tag": "select_static",
        "name": _strip_surrogates(name),
        "placeholder": {"tag": "plain_text", "content": _safe_one_line(placeholder, limit=80)},
        "options": option_items,
        "value": cb,
        "behaviors": [{"type": "callback", "value": dict(cb)}],
    }
    values = [item["value"] for item in option_items]
    selected_clean = _strip_surrogates(selected) if selected is not None else None
    if selected_clean and selected_clean in values:
        element["initial_option"] = selected_clean
    return element


def _format_member_line(
    binding: WorkflowAgentBinding,
    *,
    is_orchestrator: bool,
    is_editing: bool,
) -> str:
    role_val = binding.role or (binding.roles[0] if binding.roles else "worker")
    parts = [
        f"`{_escape_md(binding.agent_id, limit=24)}`",
        f"角色 `{_escape_md(role_val, limit=24)}`",
        f"`{_escape_md(binding.tool_name, limit=24)}`",
        f"模型 `{_escape_md(binding.model_name or 'default', limit=48)}`",
    ]
    if binding.profile:
        parts.append(f"Profile `{_escape_md(binding.profile, limit=24)}`")
    if binding.effort:
        parts.append(f"Effort `{_escape_md(binding.effort, limit=24)}`")
    badges: list[str] = []
    if is_orchestrator:
        badges.append("★ 主编排")
    if is_editing:
        badges.append("[正在编辑]")
    suffix = f" · {' '.join(badges)}" if badges else ""
    return f"- {' · '.join(parts)}{suffix}"


def _build_card_2_shell(
    *,
    title: str,
    subtitle: str | None,
    template: str,
    summary_content: str,
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    header: dict[str, Any] = {
        "title": {"tag": "plain_text", "content": _safe_one_line(title, limit=60)},
        "template": template,
    }
    if subtitle:
        header["subtitle"] = {
            "tag": "plain_text",
            "content": _safe_one_line(subtitle, limit=80),
        }
    card: dict[str, Any] = {
        "schema": "2.0",
        "config": {
            "compact_width": False,
            "wide_screen_mode": True,
            "update_multi": True,
            "summary": {"content": _strip_surrogates(summary_content)},
        },
        "header": header,
        "body": {"elements": elements},
    }
    require_card_json_2(card)
    return card


def build_workflow_config_card(
    state: WorkflowConfigEditorState | Mapping[str, Any],
    capabilities: Mapping[str, ToolCapability],
    *,
    model_options_by_tool: Mapping[str, Sequence[ACPModelOption]] | None = None,
) -> dict[str, Any]:
    """Build a Card JSON 2.0 Teamwork /wf_config team & tool+model+effort editor."""
    cfg = _coerce_state(state)

    available_caps = {
        name: cap for name, cap in capabilities.items() if cap.available
    }
    if not available_caps:
        available_caps = dict(capabilities)
    tool_names = tuple(available_caps.keys())

    members = tuple(cfg.members[:MAX_WORKFLOW_AGENT_POOL_SIZE])
    orchestrator_id = cfg.orchestrator_agent_id or (
        members[0].agent_id if members else ""
    )
    selected_agent_id = cfg.selected_agent_id
    if not selected_agent_id:
        selected_agent_id = members[0].agent_id if members else "__new__"

    editor_tool = str(cfg.editor_tool or "").strip().lower()
    if editor_tool not in available_caps:
        if members and members[0].tool_name in available_caps:
            editor_tool = members[0].tool_name
        elif tool_names:
            editor_tool = tool_names[0]

    cap = available_caps.get(editor_tool) or ToolCapability(
        tool_name=editor_tool or "codex",
        display_name=editor_tool or "Codex",
        available=True,
        supports_model=True,
        supports_effort=True,
    )

    raw_tool_options = (
        model_options_by_tool[editor_tool]
        if model_options_by_tool and editor_tool in model_options_by_tool
        else None
    )
    tool_options = normalize_workflow_acp_options(
        cap,
        raw_tool_options,
        current_model=cfg.editor_model,
    )

    (
        cascade,
        raw_resolved_model,
        resolved_model,
        raw_resolved_profile,
        resolved_profile,
        resolved_effort,
        has_adjustable_effort,
    ) = resolve_workflow_cascade_state(
        cap,
        tool_options,
        editor_model=cfg.editor_model,
        editor_profile=cfg.editor_profile,
        editor_effort=cfg.editor_effort,
    )
    raw_resolved_effort = (
        (cascade.selected_effort if cascade.efforts else None)
        if has_adjustable_effort
        else None
    )
    resolved_role = cfg.editor_role or "worker"
    valid_roles = {role_id for role_id, _ in WORKFLOW_ROLE_CATALOG}
    if resolved_role not in valid_roles:
        resolved_role = "worker"

    base_cb: dict[str, Any] = {
        "chat_id": cfg.chat_id,
        "project_id": cfg.project_id,
        "thread_root_id": cfg.thread_root_id,
        "config_session_id": cfg.config_session_id,
        "preset_id": cfg.preset_id,
        "agent_id": selected_agent_id,
        "orchestrator_agent_id": orchestrator_id or None,
        "role": resolved_role,
        "tool_name": editor_tool,
        "model_name": resolved_model,
        "profile": resolved_profile,
        "effort": resolved_effort,
    }

    def cb_for(action: str, **overrides: Any) -> dict[str, Any]:
        payload = dict(base_cb)
        payload["action"] = action
        payload.update(overrides)
        return _clean_callback_payload(payload)

    elements: list[dict[str, Any]] = []
    is_launch_card = bool(cfg.requirement)

    if cfg.status_notice:
        elements.append(
            {
                "tag": "markdown",
                "content": f"ℹ️ {_escape_md(cfg.status_notice, limit=240)}",
            }
        )

    req_text = (
        f"**任务需求**：{_escape_md(cfg.requirement, limit=240)}"
        if is_launch_card
        else "**待启动需求**：未指定任务，保存后发送 `/wf <需求>` 启动"
    )
    if cfg.saved_as_default:
        req_text += " · ✅ 已保存为项目默认"
    elements.append({"tag": "markdown", "content": req_text})

    roster_title = (
        f"**当前协作团队（{len(members)}/{MAX_WORKFLOW_AGENT_POOL_SIZE} · 预设 `{_escape_md(cfg.preset_id, limit=32)}` · 支持同模型多 Effort 共存）**"
        if is_launch_card
        else f"**当前团队成员（{len(members)}/{MAX_WORKFLOW_AGENT_POOL_SIZE} · 支持同模型多 Effort 共存）**"
    )
    roster_lines = [roster_title]
    if members:
        for member in members:
            roster_lines.append(
                _format_member_line(
                    member,
                    is_orchestrator=(member.agent_id == orchestrator_id),
                    is_editing=(member.agent_id == selected_agent_id),
                )
            )
    else:
        roster_lines.append("- 暂无成员，请在下方配置并点击「添加/更新成员」")

    preset_options = [
        (pid, f"{label} — {desc}" if len(desc) <= 22 else label)
        for pid, label, desc in WORKFLOW_PRESET_CATALOG
    ]
    agent_options: list[tuple[str, str]] = [
        (
            m.agent_id,
            f"{m.agent_id} · {m.role or 'worker'} · {m.tool_name}"
            + (f" ({m.effort})" if m.effort else ""),
        )
        for m in members
    ]
    if len(members) < MAX_WORKFLOW_AGENT_POOL_SIZE:
        agent_options.append(("__new__", "+ 新增成员"))

    if is_launch_card:
        elements.append({"tag": "markdown", "content": "\n".join(roster_lines)})
        quick_launch_buttons = [
            build_callback_button(
                text="🚀 按当前团队立即开始",
                action=cb_for(action_ids.WF_START_WORKFLOW),
                button_type="primary",
            ),
            build_callback_button(
                text="💾 保存为默认团队",
                action=cb_for(action_ids.WF_SAVE_TEAM),
                button_type="default",
            ),
        ]
        elements.extend(build_responsive_layout(quick_launch_buttons, layout="desktop"))
        elements.append({"tag": "hr"})

        # 1. Preset quick-select bar
        elements.append({"tag": "markdown", "content": "**协作预设（Preset）**"})
        elements.append(
            _select_static(
                name="preset_id",
                placeholder="选择预设模板",
                options=preset_options,
                selected=cfg.preset_id,
                callback_value=cb_for(action_ids.WF_SELECT_PRESET),
            )
        )

        # 2. Member selector
        elements.append({"tag": "markdown", "content": "**选择要编辑的成员**"})
        elements.append(
            _select_static(
                name="agent_id",
                placeholder="选择要编辑的成员",
                options=agent_options,
                selected=selected_agent_id,
                callback_value=cb_for(action_ids.WF_SELECT_AGENT),
            )
        )
        elements.append({"tag": "hr"})
    else:
        elements.append({"tag": "hr"})

        # 1. Preset quick-select bar
        elements.append({"tag": "markdown", "content": "**协作预设（Preset）**"})
        elements.append(
            _select_static(
                name="preset_id",
                placeholder="选择预设模板",
                options=preset_options,
                selected=cfg.preset_id,
                callback_value=cb_for(action_ids.WF_SELECT_PRESET),
            )
        )

        # 2. Current Team Roster (compact single Markdown + member selector)
        elements.append({"tag": "markdown", "content": "\n".join(roster_lines)})
        elements.append(
            _select_static(
                name="agent_id",
                placeholder="选择要编辑的成员",
                options=agent_options,
                selected=selected_agent_id,
                callback_value=cb_for(action_ids.WF_SELECT_AGENT),
            )
        )
        elements.append({"tag": "hr"})

    # 3. Member Editor Cascade (Role / Tool / Model / Profile / Effort)
    elements.append({"tag": "markdown", "content": "**成员配置（Role × Tool × Model × Effort）**"})
    elements.append(
        _select_static(
            name="role",
            placeholder="选择协作角色",
            options=WORKFLOW_ROLE_CATALOG,
            selected=resolved_role,
            callback_value=cb_for(action_ids.WF_SELECT_ROLE),
        )
    )

    if tool_names:
        tool_options_ui = [
            (
                t_name,
                f"{t_name} ({available_caps[t_name].display_name})"
                if available_caps[t_name].display_name.lower() != t_name
                else t_name,
            )
            for t_name in tool_names[:_MAX_SELECT_OPTIONS]
        ]
        elements.append(
            _select_static(
                name="tool_name",
                placeholder="选择编程工具",
                options=tool_options_ui,
                selected=editor_tool,
                callback_value=cb_for(action_ids.WF_SELECT_TOOL),
            )
        )

    capped_models = _cap_options_preserving(
        cascade.model_names,
        keep=("default", raw_resolved_model),
        limit=_MAX_SELECT_OPTIONS,
    ) or ("default",)
    model_ui_options = [
        ("default", "default（默认模型）") if m == "default" else (m, m)
        for m in capped_models
    ]
    elements.append(
        _select_static(
            name="model_name",
            placeholder="选择模型",
            options=model_ui_options,
            selected=raw_resolved_model,
            callback_value=cb_for(action_ids.WF_SELECT_MODEL),
        )
    )

    capped_profiles = _cap_options_preserving(
        cascade.profiles,
        keep=(raw_resolved_profile,),
        limit=_MAX_SELECT_OPTIONS,
    )
    if capped_profiles:
        elements.append(
            _select_static(
                name="profile",
                placeholder="选择 Profile",
                options=[
                    ("default", "default（默认 Profile）") if p == "default" else (p, p)
                    for p in capped_profiles
                ],
                selected=raw_resolved_profile,
                callback_value=cb_for(action_ids.WF_SELECT_PROFILE),
            )
        )

    capped_efforts = _cap_options_preserving(
        cascade.efforts,
        keep=("default", raw_resolved_effort),
        limit=_MAX_SELECT_OPTIONS,
    )
    if has_adjustable_effort and capped_efforts:
        effort_ui_options = [
            ("default", "default（默认 Effort）") if e == "default" else (e, e)
            for e in capped_efforts
        ]
        effort_selected = raw_resolved_effort or "default"
    else:
        effort_ui_options = [("default", "default（默认 Effort · 无需切换）")]
        effort_selected = "default"
    elements.append(
        _select_static(
            name="effort",
            placeholder="选择 Effort",
            options=effort_ui_options,
            selected=effort_selected,
            callback_value=cb_for(action_ids.WF_SELECT_EFFORT),
        )
    )

    editor_buttons = [
        build_callback_button(
            text="添加/更新成员",
            action=cb_for(action_ids.WF_ADD_OR_UPDATE_AGENT),
            button_type="primary",
        ),
        build_callback_button(
            text="设为编排器",
            action=cb_for(action_ids.WF_SET_ORCHESTRATOR),
        ),
        build_callback_button(
            text="移除成员",
            action=cb_for(action_ids.WF_REMOVE_AGENT),
        ),
    ]
    elements.extend(build_responsive_layout(editor_buttons, layout="desktop"))
    elements.append({"tag": "hr"})

    # 4. Footer actions
    if is_launch_card:
        footer_buttons = [
            build_callback_button(
                text="重置默认",
                action=cb_for(action_ids.WF_RESET_TEAM),
            ),
            build_callback_button(
                text="🚀 按当前团队立即开始",
                action=cb_for(action_ids.WF_START_WORKFLOW),
                button_type="primary",
            ),
        ]
    else:
        footer_buttons = [
            build_callback_button(
                text="保存为项目默认",
                action=cb_for(action_ids.WF_SAVE_TEAM),
                button_type="primary",
            ),
            build_callback_button(
                text="重置默认",
                action=cb_for(action_ids.WF_RESET_TEAM),
            ),
        ]
    elements.extend(build_responsive_layout(footer_buttons, layout="desktop"))

    summary = _fit_summary(
        f"Workflow 团队配置 · {len(members)} 成员 · {cfg.preset_id}",
        default="Workflow 团队配置与模型档位编辑器",
    )
    card_title = "⚡ Workflow · 配置团队并启动" if is_launch_card else "⚡ Workflow 团队与模型配置"
    card_subtitle = (
        "可直接按当前配置启动，或在下方灵活调整 工具 + 模型 + Effort"
        if is_launch_card
        else f"{len(members)}/{MAX_WORKFLOW_AGENT_POOL_SIZE} 成员 · 同模型多 Effort 共存"
    )
    card = _build_card_2_shell(
        title=card_title,
        subtitle=card_subtitle,
        template="blue",
        summary_content=summary,
        elements=elements,
    )
    _assert_card_within_budget(card)
    return card


def build_workflow_help_card(
    *,
    node_requirement_text: str,
    project_id: str | None = None,
    chat_id: str = "",
) -> dict[str, Any]:
    """Build a pure Card JSON 2.0 help card for ``/wf_help``."""
    req_note = _escape_md(node_requirement_text, limit=120) if node_requirement_text else "Node.js >= 20"
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                "**🚀 快速入口与命令**\n"
                "- `/wf <需求>`（`/workflow`）：打开团队与 `tool + model + effort` 交互配置卡（预填已保存/推荐团队，支持 1 键启动或灵活微调后启动）\n"
                "- `/wf --auto <需求>` / `/wf --preset <预设> <需求>`：跳过配置卡，直接按默认或指定预设团队极速启动\n"
                "- `/wf` / `/wf_config`（`/workflow_config`）：配置当前聊天+项目的默认协作团队\n"
                "- `/wf_status`（`/workflow_status`）：查看当前 Workflow 池、计划与执行进度\n"
                "- `/stop_wf`（`/stop_workflow`）：停止当前运行或生成中的 Workflow\n"
                "- `/wf_help`（`/workflow_help`）：查看本帮助"
            ),
        },
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": (
                "**🧩 Teamwork 协作预设与角色**\n"
                "- 自动绑定与保存：未配置时自动绑定推荐 Agent Pool（1–8 个成员），已保存团队按项目+会话自动沿用\n"
                "- 预设：`collaborative`（推荐协作组） · `quick`（快速双人） · `review_team`（深度复核） · `deep_collab`（深度协作） · `auto` · `custom`\n"
                "- 角色：`scout` · `planner` · `worker` · `verifier` · `synthesizer`\n"
                "- 同模型多档位：支持同一 `tool + model` 搭配不同 `effort`（如 `low` + `xhigh`）共存\n"
                f"- 运行环境要求：{req_note}"
            ),
        },
        {"tag": "hr"},
        *build_responsive_layout(
            [
                build_callback_button(
                    text="⚙️ 打开团队配置",
                    action=_clean_callback_payload(
                        {
                            "action": action_ids.WF_OPEN_CONFIG,
                            "chat_id": chat_id,
                            "project_id": project_id,
                        }
                    ),
                    button_type="primary",
                )
            ],
            layout="mobile",
        ),
    ]
    card = _build_card_2_shell(
        title="⚡ Workflow 多代理协作帮助",
        subtitle="Teamwork · tool + model + effort",
        template="blue",
        summary_content=_fit_summary(
            "Workflow 多代理协作命令与配置说明",
            default="Workflow 多代理协作帮助卡片",
        ),
        elements=elements,
    )
    _assert_card_within_budget(card)
    return card


def build_workflow_preset_launch_card(
    *,
    requirement: str,
    preset_id: str,
    members: Sequence[WorkflowAgentBinding],
    orchestrator_agent_id: str,
    chat_id: str = "",
    project_id: str | None = None,
) -> dict[str, Any]:
    """Build a pure Card JSON 2.0 quick-launch summary card for preset runs."""
    pool_lines = [
        _format_member_line(
            member,
            is_orchestrator=(member.agent_id == orchestrator_agent_id),
            is_editing=False,
        )
        for member in list(members)[:MAX_WORKFLOW_AGENT_POOL_SIZE]
    ]
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": f"**任务**\n{_escape_md(requirement, limit=300) or '未提供任务描述'}",
        },
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": (
                f"**协作预设**：`{_escape_md(preset_id, limit=40)}` · "
                f"**主编排**：`{_escape_md(orchestrator_agent_id or 'auto', limit=32)}`\n"
                + ("\n".join(pool_lines) if pool_lines else "- 自动分配可用代理")
            ),
        },
        {"tag": "hr"},
    ]
    stop_button = build_callback_button(
        text="⏹️ 停止 Workflow",
        action=_clean_callback_payload(
            {
                "action": action_ids.WORKFLOW_STOP_RUNNING,
                "chat_id": chat_id,
                "project_id": project_id,
            }
        ),
        button_type="danger",
    )
    stop_button["size"] = "medium"
    elements.extend(
        build_responsive_layout(
            [stop_button],
            layout="mobile",
        )
    )
    card = _build_card_2_shell(
        title="⚡ Workflow · 预设团队启动中",
        subtitle=f"{preset_id} · {len(members)} 成员",
        template="blue",
        summary_content=_fit_summary(
            f"Workflow 预设 {preset_id} 启动中",
            default="Workflow 预设团队启动中",
        ),
        elements=elements,
    )
    _assert_card_within_budget(card)
    return card


def _assert_card_within_budget(card: dict[str, Any]) -> None:
    nodes = count_tagged_nodes(card)
    if nodes > _CARD_NODE_BUDGET:
        raise ValueError(f"Workflow card exceeds node budget: {nodes} > {_CARD_NODE_BUDGET}")
    size = len(json.dumps(card, ensure_ascii=False).encode("utf-8", errors="strict"))
    if size > _CARD_BYTE_BUDGET:
        raise ValueError(f"Workflow card exceeds byte budget: {size} > {_CARD_BYTE_BUDGET}")


__all__ = [
    "WORKFLOW_PRESET_CATALOG",
    "WORKFLOW_ROLE_CATALOG",
    "WorkflowConfigEditorState",
    "build_workflow_config_card",
    "build_workflow_help_card",
    "build_workflow_preset_launch_card",
    "capability_to_acp_options",
    "normalize_workflow_acp_options",
    "resolve_workflow_cascade_state",
]
