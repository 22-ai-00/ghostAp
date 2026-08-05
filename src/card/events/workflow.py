"""Workflow-specific CardEvent factory functions.

Keeps workflow construction logic separate from the core CardEvent class.

NOTE: These factory functions define the typed event API for workflow lifecycle
events. They are not yet wired into the handler layer (the handler currently
uses direct callback functions instead). These will be integrated when the
event bus architecture is adopted in a future iteration. Until then they serve
as the canonical payload contract and are exercised by tests/test_workflow_payloads.py.
"""

from __future__ import annotations

from typing import Any

from .factories import CardEvent
from .payloads import (
    WorkflowAgentDonePayload,
    WorkflowAgentFailedPayload,
    WorkflowAgentStartedPayload,
    WorkflowConfirmPayload,
    WorkflowLogPayload,
    WorkflowPhasePayload,
    WorkflowProgressPayload,
)
from .types import CardEventType


def workflow_progress(
    card: dict[str, Any],
    compact_status: str = "",
) -> CardEvent:
    """Workflow progress update with full card render data.

    Payload:
        card: Feishu card JSON dict with header + elements (required).
        compact_status: One-line text summary (optional).
    Triggered when: Any agent starts/completes or phase changes.
    """
    if not isinstance(card, dict):
        raise TypeError(
            f"workflow_progress() card must be a dict, got {type(card).__name__}"
        )
    payload: WorkflowProgressPayload = {"card": card}
    if compact_status:
        payload["compact_status"] = compact_status
    return CardEvent(type=CardEventType.WORKFLOW_PROGRESS, payload=payload)


def workflow_phase(title: str) -> CardEvent:
    """Workflow phase transition.

    Payload:
        title: Phase title from the workflow script.
    Triggered when: The JS runtime calls phase(title).
    """
    payload: WorkflowPhasePayload = {"title": title}
    return CardEvent(type=CardEventType.WORKFLOW_PHASE, payload=payload)


def workflow_agent_started(label: str, tool: str, phase: str = "") -> CardEvent:
    """Workflow agent call started.

    Payload:
        label: Agent label/identifier.
        tool: Programming tool being used (e.g. "coco", "claude").
        phase: Current phase title.
    Triggered when: An agent() call begins execution.
    """
    payload: WorkflowAgentStartedPayload = {
        "label": label,
        "tool": tool,
        "phase": phase,
    }
    return CardEvent(type=CardEventType.WORKFLOW_AGENT_STARTED, payload=payload)


def workflow_agent_done(
    label: str,
    token_usage: int | None = None,
    duration_s: float | None = None,
    cached: bool | None = None,
) -> CardEvent:
    """Workflow agent call completed successfully.

    Uses explicit None-sentinel so that token_usage=0, duration_s=0.0 and
    cached=False are meaningful values written to the payload. Only omit a
    field when it was not reported at all (parameter left as None).

    Payload:
        label: Agent label/identifier (required).
        token_usage: Tokens consumed by this call (optional; 0 is valid).
        duration_s: Wall-clock duration in seconds (optional; 0.0 is valid).
        cached: Whether the result came from journal cache (optional).
    Triggered when: An agent() call completes without error.
    """
    payload: WorkflowAgentDonePayload = {"label": label}
    if token_usage is not None:
        payload["token_usage"] = token_usage
    if duration_s is not None:
        payload["duration_s"] = duration_s
    if cached is not None:
        payload["cached"] = cached
    return CardEvent(type=CardEventType.WORKFLOW_AGENT_DONE, payload=payload)


def workflow_agent_failed(label: str, error: str) -> CardEvent:
    """Workflow agent call failed.

    Payload:
        label: Agent label/identifier.
        error: Error description.
    Triggered when: An agent() call encounters an error.
    """
    payload: WorkflowAgentFailedPayload = {"label": label, "error": error}
    return CardEvent(type=CardEventType.WORKFLOW_AGENT_FAILED, payload=payload)


def workflow_log(message: str) -> CardEvent:
    """Workflow log message from the JS runtime.

    Payload:
        message: Log text.
    Triggered when: The JS runtime calls log(message).
    """
    payload: WorkflowLogPayload = {"message": message}
    return CardEvent(type=CardEventType.WORKFLOW_LOG, payload=payload)


def workflow_confirm(
    script_name: str,
    description: str,
    phases: list[dict],
    tools: list[str],
    requirement: str,
    initiator_user_id: str,
    engine_session_key: str,
    *,
    project_id: str = "",
    chat_id: str = "",
    is_fallback: bool = False,
    workflow_refs: list[dict] | None = None,
    dependency_graph: dict | None = None,
    phase_tool_mapping: dict | None = None,
    script_preview: str = "",
) -> CardEvent:
    """Workflow confirmation request — shows script preview before execution.

    Payload:
        script_name: Name from meta.name.
        description: Description from meta.description.
        phases: List of {title, detail} dicts.
        tools: Recommended tools list.
        requirement: Original user requirement.
        initiator_user_id: User who initiated (security binding).
        engine_session_key: Unique session key for callback validation.
        project_id: Project context ID.
        chat_id: Chat context ID.
        is_fallback: Whether fallback script was used.
        workflow_refs: Reference files [{name, description?, args?, failure_policy?}].
            String refs (plain template names) are supported for backward
            compatibility and are normalized to {name} dicts at the handler
            boundary. ``name`` is required; ``description`` / ``args`` /
            ``failure_policy`` are optional and control how the parent
            script injects the sub-workflow call.
        dependency_graph: Phase dependency map {phase: [dep_phases]}.
        phase_tool_mapping: Per-phase tool overrides {phase: [tools]}.
        script_preview: Optional truncated script preview shown on the confirm card.
    Triggered when: Script generation completes and awaits user confirmation.
    """
    payload: WorkflowConfirmPayload = {
        "script_name": script_name,
        "description": description,
        "phases": phases,
        "tools": tools,
        "requirement": requirement,
        "initiator_user_id": initiator_user_id,
        "engine_session_key": engine_session_key,
    }
    if project_id:
        payload["project_id"] = project_id
    if chat_id:
        payload["chat_id"] = chat_id
    if is_fallback:
        payload["is_fallback"] = is_fallback
    if workflow_refs is not None:
        payload["workflow_refs"] = workflow_refs
    if dependency_graph is not None:
        payload["dependency_graph"] = dependency_graph
    if phase_tool_mapping is not None:
        payload["phase_tool_mapping"] = phase_tool_mapping
    if script_preview:
        payload["script_preview"] = script_preview
    return CardEvent(type=CardEventType.WORKFLOW_CONFIRM, payload=payload)
