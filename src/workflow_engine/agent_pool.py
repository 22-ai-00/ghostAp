"""Immutable identities for user-confirmed Workflow Agent bindings."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_EFFORT_CAPABLE_TOOLS = frozenset(
    {"codex", "codex_w", "claude", "claude_w", "agy", "traex", "dsh"}
)


def _normalize_roles(raw_roles: object) -> tuple[str, ...]:
    if raw_roles is None:
        return ()
    if isinstance(raw_roles, str):
        items: Iterable[object] = (raw_roles,)
    elif isinstance(raw_roles, Iterable):
        items = raw_roles
    else:
        return ()
    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _cached_default_model(tool_name: str, *, skip_orchestrator: bool = False) -> str | None:
    try:
        from src.acp.helper import get_cached_acp_models

        options = get_cached_acp_models(tool_name)
    except Exception:
        return None
    if not options:
        return None
    if skip_orchestrator:
        from src.acp.agy_selection import is_agy_orchestrator_model

        options = [opt for opt in options if not is_agy_orchestrator_model(opt.name)]
        if not options:
            return None
    default_opt = next((opt for opt in options if opt.is_default), None) or options[0]
    name = str(getattr(default_opt, "name", "") or "").strip()
    return name or None


def resolve_effective_model_name(
    tool_name: str,
    model_name: str | None = None,
    *,
    profile: str | None = None,
    effort: str | None = None,
) -> str | None:
    """Resolve the canonical backend ``model_name`` string for a tool binding.

    Idempotent: calling this function on its own return value produces the
    identical canonical string. Model-only backends (``coco``, ``aiden``,
    ``gemini``, ``grok``) return ``model_name`` verbatim without slash
    splitting so provider-qualified IDs remain intact.
    """
    tool = str(tool_name or "").strip().lower()
    raw_model = str(model_name or "").strip() or None
    prof = str(profile or "").strip().lower() or None
    eff = str(effort or "").strip().lower() or None

    if tool not in _EFFORT_CAPABLE_TOOLS:
        return raw_model

    if tool in {"claude", "claude_w"}:
        from src.acp.claude_selection import (
            CLAUDE_DEFAULT_EFFORT_TOKEN,
            CLAUDE_DEFAULT_MODEL_TOKEN,
            compose_claude_model_selection,
            split_claude_model_selection,
        )

        emb_base, emb_effort = split_claude_model_selection(raw_model)
        resolved_effort = eff or emb_effort
        if resolved_effort == CLAUDE_DEFAULT_EFFORT_TOKEN:
            resolved_effort = None
        base = emb_base or (CLAUDE_DEFAULT_MODEL_TOKEN if resolved_effort else None)
        if base is None:
            return None
        return compose_claude_model_selection(base, resolved_effort)

    if tool in {"codex", "codex_w"}:
        from src.acp.model_selection import (
            CODEX_DEFAULT_EFFORT_TOKEN,
            CODEX_DEFAULT_MODEL_TOKEN,
            compose_codex_model_selection,
            split_codex_model_selection,
        )

        emb_base, emb_effort = split_codex_model_selection(raw_model)
        resolved_effort = eff or emb_effort
        if resolved_effort == CODEX_DEFAULT_EFFORT_TOKEN:
            resolved_effort = None
        base = emb_base
        if base is None and resolved_effort:
            if tool == "codex":
                base = _cached_default_model("codex") or CODEX_DEFAULT_MODEL_TOKEN
            else:
                base = CODEX_DEFAULT_MODEL_TOKEN
        if base is None:
            return None
        return compose_codex_model_selection(base, resolved_effort)

    if tool == "traex":
        from src.acp.traex_selection import (
            compose_traex_model_selection,
            load_traex_model_metadata,
            split_traex_model_selection,
        )

        emb_base, emb_profile, emb_effort = split_traex_model_selection(raw_model)
        resolved_profile = prof or emb_profile or "standard"
        resolved_effort = eff or emb_effort
        base = emb_base or None
        if base is None and (resolved_effort or resolved_profile != "standard"):
            meta = load_traex_model_metadata()
            if meta:
                base = meta[0].config_name
            else:
                base = _cached_default_model("traex") or "default"
        if base is None:
            return None
        return compose_traex_model_selection(base, resolved_profile, resolved_effort)

    if tool == "agy":
        from src.acp.agy_selection import (
            AGY_ORCHESTRATOR_DEFAULT_PROFILE,
            AGY_REASONING_EFFORTS,
            compose_agy_model_selection,
            compose_agy_orchestrator_selection,
            is_agy_orchestrator_model,
            parse_agy_model_selection,
        )
        from src.acp.model_selection import split_slash_model_effort

        if raw_model and is_agy_orchestrator_model(raw_model):
            try:
                parsed = parse_agy_model_selection(raw_model)
            except ValueError:
                return raw_model
            planner = prof or parsed.planner_effort
            executor = eff or parsed.executor_effort
            if (
                executor
                and executor != AGY_ORCHESTRATOR_DEFAULT_PROFILE
                and (not planner or planner == AGY_ORCHESTRATOR_DEFAULT_PROFILE)
            ):
                planner = executor
            return compose_agy_orchestrator_selection(
                parsed.model_id,
                planner,
                executor,
            )

        if raw_model:
            model_id, thought_level = split_slash_model_effort(
                raw_model,
                allowed_efforts=AGY_REASONING_EFFORTS,
            )
        else:
            model_id, thought_level = None, None
        resolved_effort = eff or thought_level
        if resolved_effort == AGY_ORCHESTRATOR_DEFAULT_PROFILE:
            resolved_effort = None
        base = model_id
        if base is None and resolved_effort:
            base = _cached_default_model("agy", skip_orchestrator=True) or "default"
        if base is None:
            return None
        if resolved_effort and resolved_effort in AGY_REASONING_EFFORTS:
            return compose_agy_model_selection(base, resolved_effort)
        return base

    if tool == "dsh":
        from src.acp.dsh_selection import (
            compose_dsh_model_selection,
            decode_dsh_reasoning_value,
            split_dsh_model_selection,
        )

        if raw_model:
            if raw_model.lower() in {"default", "default/default"}:
                if eff:
                    return json.dumps(
                        ["default", "default", eff],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                return "default"
            if raw_model.startswith("["):
                try:
                    model_json, reasoning_json = split_dsh_model_selection(raw_model)
                except ValueError:
                    return raw_model
                existing_effort = (
                    decode_dsh_reasoning_value(reasoning_json) if reasoning_json else None
                )
                return compose_dsh_model_selection(model_json, eff or existing_effort)
            if "/" in raw_model:
                provider, _, model_id = raw_model.partition("/")
                if provider.strip() and model_id.strip():
                    model_json = json.dumps(
                        [provider.strip(), model_id.strip()],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    return compose_dsh_model_selection(model_json, eff)
            return raw_model
        if eff:
            cached = _cached_default_model("dsh")
            if cached and "/" in cached:
                provider, _, model_id = cached.partition("/")
                if provider.strip() and model_id.strip():
                    model_json = json.dumps(
                        [provider.strip(), model_id.strip()],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    return compose_dsh_model_selection(model_json, eff)
            return json.dumps(
                ["default", "default", eff],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return None

    return raw_model


@dataclass(frozen=True, slots=True)
class WorkflowAgentBinding:
    """One stable Agent identity bound to an exact backend tool, model, and effort."""

    agent_id: str
    tool_name: str
    model_name: str | None
    display_name: str
    profile: str | None = None
    effort: str | None = None
    role: str | None = None
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        agent_id = str(self.agent_id or "").strip()
        tool_name = str(self.tool_name or "").strip().lower()
        model_name = str(self.model_name or "").strip() or None
        display_name = str(self.display_name or "").strip()
        profile = str(self.profile or "").strip().lower() or None
        effort = str(self.effort or "").strip().lower() or None
        role = str(self.role or "").strip().lower() or None
        roles = _normalize_roles(self.roles)
        if not _AGENT_ID_RE.fullmatch(agent_id):
            raise ValueError(
                "Workflow Agent agent_id must be 1-128 safe identifier characters"
            )
        if not tool_name:
            raise ValueError("Workflow Agent tool_name must not be empty")
        if not display_name:
            raise ValueError("Workflow Agent display_name must not be empty")
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "effort", effort)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "roles", roles)

    @property
    def binding_key(self) -> tuple[str, str, str, str]:
        return (
            self.tool_name,
            self.model_name or "",
            self.profile or "",
            self.effort or "",
        )

    @property
    def tool_model_key(self) -> tuple[str, str, str, str]:
        return self.binding_key

    def effective_model_name(self) -> str | None:
        return resolve_effective_model_name(
            self.tool_name,
            self.model_name,
            profile=self.profile,
            effort=self.effort,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "profile": self.profile,
            "effort": self.effort,
            "role": self.role,
            "roles": list(self.roles),
        }

    @classmethod
    def from_dict(cls, data: object) -> "WorkflowAgentBinding | None":
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                agent_id=str(data.get("agent_id") or data.get("agentId") or ""),
                tool_name=str(data.get("tool_name") or data.get("toolName") or ""),
                model_name=(
                    str(data.get("model_name") or data.get("modelName") or "").strip()
                    or None
                ),
                display_name=str(
                    data.get("display_name")
                    or data.get("displayName")
                    or data.get("agent_id")
                    or data.get("agentId")
                    or ""
                ),
                profile=str(data.get("profile") or "").strip().lower() or None,
                effort=str(data.get("effort") or "").strip().lower() or None,
                role=str(data.get("role") or "").strip().lower() or None,
                roles=_normalize_roles(data.get("roles")),
            )
        except ValueError:
            return None


def freeze_agent_pool(
    bindings: Iterable[WorkflowAgentBinding | dict[str, Any]],
    *,
    require_nonempty: bool = True,
) -> tuple[WorkflowAgentBinding, ...]:
    """Normalize an ordered pool and reject ambiguous identities or bindings."""

    pool: list[WorkflowAgentBinding] = []
    for raw in bindings:
        binding = raw if isinstance(raw, WorkflowAgentBinding) else WorkflowAgentBinding.from_dict(raw)
        if binding is None:
            raise ValueError("Workflow agent_pool contains an invalid binding")
        pool.append(binding)
    if require_nonempty and not pool:
        raise ValueError("Workflow agent_pool must not be empty")

    seen_ids: set[str] = set()
    seen_tool_models: set[tuple[str, str, str, str]] = set()
    for binding in pool:
        if binding.agent_id in seen_ids:
            raise ValueError(f"Workflow agent_pool has duplicate agent_id: {binding.agent_id}")
        if binding.binding_key in seen_tool_models:
            tool, model, profile, effort = binding.binding_key
            detail = "/".join(part for part in (model or "default", profile, effort) if part)
            raise ValueError(
                "Workflow agent_pool has duplicate tool/model binding: "
                f"{tool}/{detail}"
            )
        seen_ids.add(binding.agent_id)
        seen_tool_models.add(binding.binding_key)
    return tuple(pool)


def select_auto_orchestrator(
    bindings: Iterable[WorkflowAgentBinding | dict[str, Any]],
    *,
    recommendations: Sequence[dict[str, Any]] = (),
    preferred_tools: Sequence[str] = (
        "traex",
        "claude",
        "codex",
        "grok",
        "dsh",
        "agy",
        "aiden",
        "gemini",
        "coco",
    ),
) -> WorkflowAgentBinding:
    """Resolve Auto independently of pool insertion order."""
    pool = freeze_agent_pool(bindings)
    exact_4tuple_rank: dict[tuple[str, str, str, str], int] = {}
    exact_2tuple_rank: dict[tuple[str, str | None], int] = {}
    recommended_tool_rank: dict[str, int] = {}
    for index, raw in enumerate(recommendations):
        tool = str(raw.get("tool_name") or raw.get("toolName") or "").strip().lower()
        if not tool:
            continue
        model = str(raw.get("model_name") or raw.get("modelName") or "").strip() or None
        profile = str(raw.get("profile") or "").strip().lower()
        effort = str(raw.get("effort") or "").strip().lower()
        exact_4tuple_rank.setdefault((tool, model or "", profile, effort), index)
        exact_2tuple_rank.setdefault((tool, model), index)
        recommended_tool_rank.setdefault(tool, index)
    preferred_rank = {
        str(tool).strip().lower(): index for index, tool in enumerate(preferred_tools)
    }

    def agent_id_rank(agent_id: str) -> tuple[int, str]:
        match = re.fullmatch(r"A-?(\d+)", agent_id, re.IGNORECASE)
        return (int(match.group(1)), "") if match else (10**9, agent_id)

    sentinel = 10**9
    return min(
        pool,
        key=lambda binding: (
            exact_4tuple_rank.get(binding.binding_key, sentinel),
            exact_2tuple_rank.get((binding.tool_name, binding.model_name), sentinel),
            recommended_tool_rank.get(binding.tool_name, sentinel),
            preferred_rank.get(binding.tool_name, sentinel),
            0 if binding.model_name is None else 1,
            binding.model_name or "",
            agent_id_rank(binding.agent_id),
        ),
    )
