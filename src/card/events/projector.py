"""Pure ACP subagent event projections shared by card lifecycles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from src.card.tool_display import (
    collect_subagent_opaque_ids,
    extract_agent_tool_name,
    extract_tool_call_label,
    is_unhelpful_display_label,
    sanitize_subagent_display_text,
    sanitize_tool_failure_detail,
)

AGENT_TOOL_TITLES = frozenset({"agent", "subagent", "task"})
GENERIC_TASK_LABELS = frozenset({"", "agent", "subagent", "task", "子任务"})
TERMINAL_AGENT_STATUSES = frozenset({"completed", "failed", "cancelled"})
UNRESOLVED_AGENT_STATUS = "unresolved"
UNRESOLVED_AGENT_PROGRESS = "未收到最终状态，主任务已结束"
GENERIC_AGENT_FAILURE_DETAIL = "子任务执行失败"

_PROVIDER_STATUS = {
    "pendinginit": "running",
    "pending_init": "running",
    "pending": "running",
    "running": "running",
    "completed": "completed",
    "cancelled": "cancelled",
    "errored": "failed",
    "failed": "failed",
    "interrupted": "cancelled",
    "shutdown": "completed",
    "notfound": "failed",
    "not_found": "failed",
}
_PROVIDER_PROGRESS = {
    "pendinginit": "准备中",
    "pending_init": "准备中",
    "pending": "准备中",
    "running": "执行中",
    "completed": "已完成",
    "cancelled": "已取消",
    "errored": "执行失败",
    "failed": "执行失败",
    "interrupted": "已中断",
    "shutdown": "已完成并停止",
    "notfound": "状态不可用",
    "not_found": "状态不可用",
}
_TERMINAL_PROGRESS = {
    "completed": "已完成",
    "failed": "执行失败",
    "cancelled": "已取消",
}
_TRANSIENT_PROGRESS = frozenset({
    "准备中",
    "执行中",
    "已启动",
    "正在启动",
    "启动未完成",
    "已与主 Agent 交互",
    "正在与主 Agent 交互",
    "交互未完成",
    "正在中断",
    "中断未完成",
    "动态已更新",
})
_ACTIVITY_PROGRESS = {
    "started": ("正在启动", "已启动", "启动未完成"),
    "interacted": ("正在与主 Agent 交互", "已与主 Agent 交互", "交互未完成"),
    "interrupted": ("正在中断", "已中断", "中断未完成"),
}


@dataclass(frozen=True)
class ProjectedAgent:
    source_id: str
    label: str
    status: str
    progress: str
    model: str = ""


@dataclass(frozen=True)
class CollaborationProjection:
    agents: tuple[ProjectedAgent, ...]
    starts_new_generation: bool
    authoritative_list_snapshot: bool
    failed_without_receiver: bool


@dataclass(frozen=True)
class ActivityProjection:
    source_id: str
    label: str
    status: str
    progress: str
    interrupted: bool


def is_agent_task(tool_call: Any) -> bool:
    if tool_call is None:
        return False
    title = str(getattr(tool_call, "title", "") or "").strip().lower()
    kind = str(getattr(tool_call, "kind", "") or "").strip().lower()
    content = str(getattr(tool_call, "content", "") or "").strip()
    if str(getattr(tool_call, "subagent_source_id", "") or "").strip():
        return True
    if str(getattr(tool_call, "collaboration_tool", "") or "").strip():
        return True
    if kind == "agent" or title in AGENT_TOOL_TITLES:
        return True
    return kind == "other" and "子代理：" in content


def is_task_tool(tool_call: Any) -> bool:
    return str(getattr(tool_call, "title", "") or "").strip().lower() == "task"


def is_generic_task_label(value: object) -> bool:
    return str(value or "").strip().lower() in GENERIC_TASK_LABELS


def _raw_metadata_limit(tool_call: Any, minimum: int) -> int:
    return max(
        minimum,
        len(str(getattr(tool_call, "id", "") or "")),
        len(str(getattr(tool_call, "title", "") or "")),
        len(str(getattr(tool_call, "content", "") or "")),
    )


def extract_agent_task_label(tool_call: Any) -> str:
    opaque_ids = collect_subagent_opaque_ids(tool_call)
    path = str(getattr(tool_call, "subagent_path", "") or "").strip()
    if path:
        leaf = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if leaf:
            return sanitize_subagent_display_text(
                leaf,
                fallback="子任务",
                max_chars=60,
                opaque_ids=opaque_ids,
            )
    return sanitize_subagent_display_text(
        extract_tool_call_label(
            tool_call,
            generic_labels=GENERIC_TASK_LABELS,
            fallback="子任务",
            max_chars=_raw_metadata_limit(tool_call, 60),
        ),
        fallback="子任务",
        max_chars=60,
        opaque_ids=opaque_ids,
    )


def extract_agent_display_tool(tool_call: Any) -> str:
    return sanitize_subagent_display_text(
        extract_agent_tool_name(
            tool_call,
            max_chars=_raw_metadata_limit(tool_call, 24),
        ),
        fallback="子代理",
        max_chars=24,
        opaque_ids=collect_subagent_opaque_ids(tool_call),
    )


def agent_failure_detail(
    tool_call: Any,
    *,
    fallback: str = GENERIC_AGENT_FAILURE_DETAIL,
) -> str:
    opaque_ids = (getattr(tool_call, "id", ""),)
    result = getattr(tool_call, "result", None)
    if result is not None:
        detail = sanitize_tool_failure_detail(
            result,
            fallback="",
            opaque_ids=opaque_ids,
            allow_unstructured=False,
        )
        if detail:
            return detail
    return sanitize_tool_failure_detail(
        getattr(tool_call, "content", ""),
        fallback=fallback,
        opaque_ids=opaque_ids,
        allow_unstructured=False,
    )


def terminal_agent_label(tool_call: Any) -> str:
    title = str(getattr(tool_call, "title", "") or "").strip()
    if is_generic_task_label(title) or is_unhelpful_display_label(title):
        return "子任务"
    return sanitize_tool_failure_detail(
        title,
        fallback="子任务",
        max_chars=60,
        opaque_ids=(getattr(tool_call, "id", ""),),
    )


def project_collaboration(acp_event: Any) -> CollaborationProjection | None:
    tool_call = getattr(acp_event, "tool_call", None)
    collaboration_tool = str(
        getattr(tool_call, "collaboration_tool", "") or ""
    ).strip().casefold()
    if tool_call is None or not collaboration_tool:
        return None

    event_name = str(
        getattr(getattr(acp_event, "event_type", None), "name", "") or ""
    )
    outer_status = str(getattr(tool_call, "status", "") or "").strip().casefold()
    authoritative = (
        event_name == "TOOL_CALL_DONE"
        and collaboration_tool == "list_agents"
        and outer_status == "completed"
    )
    starts_new_generation = (
        event_name == "TOOL_CALL_DONE"
        and collaboration_tool in {"spawn_agent", "followup_task"}
        and outer_status == "completed"
    )
    states = {
        str(item.get("source_id") or "").strip(): item
        for item in getattr(tool_call, "subagent_states", ())
        if isinstance(item, Mapping)
        and str(item.get("source_id") or "").strip()
    }
    source_ids = [
        str(value or "").strip()
        for value in getattr(tool_call, "collaboration_receivers", ())
        if str(value or "").strip()
    ]
    source_ids.extend(key for key in states if key not in source_ids)
    opaque_ids = collect_subagent_opaque_ids(tool_call)
    label = extract_agent_task_label(tool_call)
    model = str(getattr(tool_call, "collaboration_model", "") or "").strip()
    agents: list[ProjectedAgent] = []
    for source_id in source_ids:
        state = states.get(source_id, {})
        raw_status = str(state.get("status") or "running").strip().lower()
        status = _PROVIDER_STATUS.get(raw_status, "running")
        default_progress = _PROVIDER_PROGRESS.get(raw_status, "执行中")
        message = sanitize_subagent_display_text(
            state.get("message"),
            fallback="",
            max_chars=180,
            opaque_ids=opaque_ids,
        )
        if status in TERMINAL_AGENT_STATUSES and message in _TRANSIENT_PROGRESS:
            message = ""
        agents.append(ProjectedAgent(
            source_id=source_id,
            label=label,
            status=status,
            progress=message or default_progress,
            model=model,
        ))
    return CollaborationProjection(
        agents=tuple(agents),
        starts_new_generation=starts_new_generation,
        authoritative_list_snapshot=authoritative,
        failed_without_receiver=not agents and outer_status == "failed",
    )


def project_activity(acp_event: Any) -> ActivityProjection | None:
    tool_call = getattr(acp_event, "tool_call", None)
    source_id = str(
        getattr(tool_call, "subagent_source_id", None)
        or getattr(acp_event, "source_id", None)
        or ""
    ).strip()
    activity = str(getattr(tool_call, "subagent_activity", "") or "").strip().lower()
    if tool_call is None or not source_id or not activity:
        return None
    provider_status = str(getattr(tool_call, "status", "") or "").lower()
    running, completed, failed = _ACTIVITY_PROGRESS.get(
        activity,
        ("动态已更新", "动态已更新", "动态已更新"),
    )
    progress = completed if provider_status == "completed" else failed if provider_status == "failed" else running
    interrupted = activity == "interrupted" and provider_status == "completed"
    return ActivityProjection(
        source_id=source_id,
        label=extract_agent_task_label(tool_call),
        status="cancelled" if interrupted else "running",
        progress=progress,
        interrupted=interrupted,
    )


def _next_sequence(summaries: Mapping[str, Mapping], parent_sequence: object) -> str:
    return f"{parent_sequence}.{chr(ord('a') + len(summaries))}"


def merge_collaboration_summaries(
    summaries: Mapping[str, Mapping],
    projection: CollaborationProjection,
    *,
    parent_sequence: object,
    base_model: str = "",
) -> dict[str, dict]:
    result = {key: dict(value) for key, value in summaries.items()}
    for agent in projection.agents:
        existing = result.get(agent.source_id, {})
        existing_status = str(existing.get("status") or "").strip().lower()
        if existing_status in TERMINAL_AGENT_STATUSES:
            if projection.starts_new_generation:
                pass
            elif agent.status != existing_status:
                if not (
                    projection.authoritative_list_snapshot
                    and agent.status in TERMINAL_AGENT_STATUSES
                ):
                    continue
            elif not agent.progress or agent.progress == str(existing.get("progress") or ""):
                continue
        label = str(existing.get("label") or agent.label).strip()
        if is_generic_task_label(label):
            label = "子任务"
        summary = {
            **existing,
            "label": label,
            "tool": str(existing.get("tool") or "子代理"),
            "status": agent.status,
            "progress": agent.progress,
        }
        if agent.status != "failed":
            summary.pop("error", None)
        summary.setdefault("sequence", _next_sequence(result, parent_sequence))
        model = agent.model or str(existing.get("model") or base_model).strip()
        if model:
            summary["model"] = model
        result[agent.source_id] = summary
    return result


def merge_activity_summary(
    summaries: Mapping[str, Mapping],
    projection: ActivityProjection,
    *,
    parent_sequence: object,
    base_model: str = "",
) -> dict[str, dict]:
    existing = dict(summaries.get(projection.source_id, {}))
    if str(existing.get("status") or "").strip().lower() in TERMINAL_AGENT_STATUSES:
        return dict(summaries)
    label = str(existing.get("label") or projection.label).strip()
    if is_generic_task_label(label):
        label = "子任务"
    summary = {
        **existing,
        "label": label,
        "tool": str(existing.get("tool") or "子代理"),
        "status": projection.status,
        "progress": projection.progress,
    }
    summary.pop("error", None)
    summary.setdefault("sequence", _next_sequence(summaries, parent_sequence))
    if base_model:
        summary.setdefault("model", base_model)
    result = {key: dict(value) for key, value in summaries.items()}
    result[projection.source_id] = summary
    return result


def merge_agent_tool_summary(
    summaries: Mapping[str, Mapping],
    acp_event: Any,
    *,
    parent_sequence: object,
    base_model: str = "",
) -> tuple[bool, dict[str, dict]]:
    tool_call = getattr(acp_event, "tool_call", None)
    tool_id = str(getattr(tool_call, "id", "") or "")
    if tool_call is None or (tool_id not in summaries and not is_agent_task(tool_call)):
        return False, dict(summaries)
    event_name = str(getattr(getattr(acp_event, "event_type", None), "name", "") or "")
    incoming = (
        "failed"
        if event_name == "TOOL_CALL_DONE"
        and str(getattr(tool_call, "status", "") or "").strip().lower() == "failed"
        else "completed" if event_name == "TOOL_CALL_DONE" else "running"
    )
    existing = dict(summaries.get(tool_id, {}))
    existing_status = str(existing.get("status") or "").strip().lower()
    if existing_status in TERMINAL_AGENT_STATUSES:
        if existing_status == incoming == "failed":
            current = str(existing.get("error") or "").strip()
            if not current or current == GENERIC_AGENT_FAILURE_DETAIL:
                detail = agent_failure_detail(tool_call, fallback="")
                if detail and detail != current:
                    existing["error"] = detail
                    result = {key: dict(value) for key, value in summaries.items()}
                    result[tool_id] = existing
                    return True, result
        return True, dict(summaries)
    terminal = incoming in TERMINAL_AGENT_STATUSES
    previous_label = str(existing.get("label") or "").strip()
    label = previous_label or terminal_agent_label(tool_call) if terminal else extract_agent_task_label(tool_call)
    if (
        not terminal
        and previous_label
        and not is_generic_task_label(previous_label)
        and not is_unhelpful_display_label(previous_label)
        and (is_generic_task_label(label) or is_unhelpful_display_label(label))
    ):
        label = previous_label
    tool_name = str(existing.get("tool") or "") if terminal else ""
    summary = {
        **existing,
        "label": label,
        "tool": tool_name or extract_agent_display_tool(tool_call),
        "status": incoming,
    }
    if incoming == "failed":
        summary["error"] = agent_failure_detail(tool_call)
    else:
        summary.pop("error", None)
    summary.setdefault("sequence", _next_sequence(summaries, parent_sequence))
    if base_model:
        summary.setdefault("model", base_model)
    result = {key: dict(value) for key, value in summaries.items()}
    result[tool_id] = summary
    return True, result


def finalize_summaries(
    summaries: Mapping[str, Mapping],
    terminal_status: str,
) -> dict[str, dict]:
    """Settle child summaries without fabricating a provider terminal state.

    ``terminal_status`` describes why the parent card is closing; it is not
    evidence that a still-live child failed or was cancelled.  Only provider
    terminal frames may assign those statuses.  Unreconciled children receive
    a distinct, non-running display status so the parent can close truthfully.
    """
    result = {key: dict(value) for key, value in summaries.items()}
    parent_is_terminal = terminal_status in TERMINAL_AGENT_STATUSES
    for source_id, existing in result.items():
        status = str(existing.get("status") or "").strip().lower()
        if status in TERMINAL_AGENT_STATUSES:
            fallback = _TERMINAL_PROGRESS.get(status)
            if fallback and existing.get("progress") in _TRANSIENT_PROGRESS:
                existing["progress"] = fallback
            continue
        existing["status"] = UNRESOLVED_AGENT_STATUS
        existing["progress"] = (
            UNRESOLVED_AGENT_PROGRESS
            if parent_is_terminal
            else "未收到最终状态"
        )
        existing.pop("error", None)
    return result


def attribute_subagent_image(acp_event: Any, label: str) -> Any:
    image = getattr(acp_event, "image", None)
    label = str(label or "").strip()
    if image is None or not label or label in str(getattr(image, "name", "")):
        return acp_event
    return replace(
        acp_event,
        image=replace(image, name=f"{label} · {image.name}"[:120]),
    )
