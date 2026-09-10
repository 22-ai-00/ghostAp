"""Provider selection value used by projected employee ACP sessions."""

from __future__ import annotations

from .claude_selection import (
    CLAUDE_REASONING_EFFORTS,
    compose_claude_model_selection,
    split_claude_model_selection,
)
from .model_selection import (
    CODEX_REASONING_EFFORTS,
    compose_codex_model_selection,
    split_codex_model_selection,
)
from .traex_selection import compose_traex_model_selection, split_traex_model_selection


def validate_employee_model_components(
    tool: str,
    model: str,
    profile: str,
    effort: str,
) -> None:
    """Validate the typed persistence contract before anchoring a hire."""
    normalized_tool = str(tool or "").strip().casefold()
    selected_model = str(model or "").strip()
    if normalized_tool == "codex":
        _, embedded_effort = split_codex_model_selection(selected_model)
        if embedded_effort:
            raise ValueError("Codex model must not include an effort suffix")
    elif normalized_tool in {"claude", "claude_w"}:
        _, embedded_effort = split_claude_model_selection(selected_model)
        if embedded_effort:
            raise ValueError("Claude model must not include an effort suffix")
    elif normalized_tool in {"traex", "trae"}:
        base_model, embedded_profile, embedded_effort = split_traex_model_selection(selected_model)
        if base_model != selected_model or embedded_profile != "standard" or embedded_effort:
            raise ValueError("Traex model must not include profile or effort suffixes")
    compose_employee_model_selection(tool, model, profile, effort)


def compose_employee_model_selection(
    tool: str,
    model: str,
    profile: str,
    effort: str,
) -> str:
    """Compose the runtime model value using existing ACP normalization."""

    normalized_tool = str(tool or "").strip().casefold()
    selected_model = str(model or "").strip()
    selected_profile = str(profile or "standard").strip().casefold()
    selected_effort = str(effort or "").strip().casefold()
    if normalized_tool == "codex":
        if selected_profile not in {"", "default", "standard"}:
            raise ValueError("Codex ACP does not support employee profiles")
        if selected_effort not in {"", "default", *CODEX_REASONING_EFFORTS}:
            raise ValueError("unsupported Codex effort")
        base_model, embedded_effort = split_codex_model_selection(selected_model)
        requested_effort = None if selected_effort in {"", "default"} else selected_effort
        if embedded_effort and embedded_effort != requested_effort:
            raise ValueError("conflicting Codex effort selection")
        return compose_codex_model_selection(
            base_model or "",
            requested_effort or embedded_effort,
        )
    if normalized_tool in {"claude", "claude_w"}:
        if selected_profile not in {"", "default", "standard"}:
            raise ValueError("Claude ACP does not support employee profiles")
        if selected_effort not in {"", "default", *CLAUDE_REASONING_EFFORTS}:
            raise ValueError("unsupported Claude effort")
        base_model, embedded_effort = split_claude_model_selection(selected_model)
        requested_effort = None if selected_effort in {"", "default"} else selected_effort
        if embedded_effort and embedded_effort != requested_effort:
            raise ValueError("conflicting Claude effort selection")
        return compose_claude_model_selection(
            base_model or "",
            requested_effort or embedded_effort,
        )
    if normalized_tool in {"traex", "trae"}:
        if selected_profile not in {"", "default", "standard", "max"}:
            raise ValueError("unsupported Traex profile")
        if selected_effort not in {"", "default", *CODEX_REASONING_EFFORTS}:
            raise ValueError("unsupported Traex effort")
        base_model, embedded_profile, embedded_effort = split_traex_model_selection(selected_model)
        requested_effort = None if selected_effort in {"", "default"} else selected_effort
        requested_profile = "standard" if selected_profile in {"", "default"} else selected_profile
        if embedded_profile != "standard" and embedded_profile != requested_profile:
            raise ValueError("conflicting Traex profile selection")
        if embedded_effort and embedded_effort != requested_effort:
            raise ValueError("conflicting Traex effort selection")
        effective_profile = embedded_profile if embedded_profile != "standard" else requested_profile
        return compose_traex_model_selection(
            base_model,
            effective_profile,
            requested_effort or embedded_effort,
        )
    if selected_profile != "standard" or selected_effort not in {"", "default"}:
        raise ValueError("backend does not support employee profile or effort")
    return selected_model


__all__ = ["compose_employee_model_selection", "validate_employee_model_components"]
