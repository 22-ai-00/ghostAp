"""Pure ACP subagent event projections shared by card lifecycles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from src.acp.child_lifecycle import (
    CHILD_GENERATION_ACTIONS,
    KNOWN_CHILD_ACTIVITIES,
    TERMINAL_CHILD_STATUSES,
    TRANSIENT_CHILD_STATUSES,
    ChildLifecycleEntry,
    ChildObservationProvenance,
    parse_child_tool_call,
    reduce_child_entry,
    strict_source_id,
)
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
TERMINAL_AGENT_STATUSES = TERMINAL_CHILD_STATUSES
UNRESOLVED_AGENT_STATUS = "unresolved"
UNRESOLVED_AGENT_PROGRESS = "未收到最终状态，主任务已结束"
GENERIC_AGENT_FAILURE_DETAIL = "子任务执行失败"

_PROVIDER_PROGRESS = {
    "pendinginit": "准备中",
    "pending_init": "准备中",
    "pending": "准备中",
    "running": "执行中",
    "in_progress": "执行中",
    "completed": "已完成",
    "cancelled": "已取消",
    "canceled": "已取消",
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
    raw_status: str = ""
    malformed: bool = False
    provenance: ChildObservationProvenance = (
        ChildObservationProvenance.COLLABORATION_SNAPSHOT
    )


@dataclass(frozen=True)
class CollaborationProjection:
    agents: tuple[ProjectedAgent, ...]
    starts_new_generation: bool
    authoritative_list_snapshot: bool
    failed_without_receiver: bool
    call_id: str = ""
    reopens_transient: bool = False
    malformed: bool = False
    container_malformed: bool = False


@dataclass(frozen=True)
class ActivityProjection:
    source_id: str
    label: str
    status: str
    progress: str
    interrupted: bool
    malformed: bool = False
    reopens: bool = False


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
    parsed = parse_child_tool_call(tool_call, event_name=event_name)
    opaque_ids = collect_subagent_opaque_ids(tool_call)
    label = extract_agent_task_label(tool_call)
    model = str(getattr(tool_call, "collaboration_model", "") or "").strip()
    agents: list[ProjectedAgent] = []
    for obs in parsed.observations:
        if obs.malformed or obs.status is None:
            status = "malformed"
            default_progress = "状态未知"
        else:
            status = obs.status
            default_progress = _PROVIDER_PROGRESS.get(
                obs.raw_status,
                _PROVIDER_PROGRESS.get(status, "执行中"),
            )
        message = sanitize_subagent_display_text(
            obs.message,
            fallback="",
            max_chars=180,
            opaque_ids=opaque_ids,
        )
        if status in TERMINAL_AGENT_STATUSES and message in _TRANSIENT_PROGRESS:
            message = ""
        agents.append(
            ProjectedAgent(
                source_id=obs.source_id,
                label=label,
                status=status,
                progress=message or default_progress,
                model=model,
                raw_status=obs.raw_status,
                malformed=obs.malformed or obs.status is None,
                provenance=obs.provenance,
            )
        )
    return CollaborationProjection(
        agents=tuple(agents),
        starts_new_generation=parsed.starts_new_generation,
        authoritative_list_snapshot=parsed.authoritative_list_snapshot,
        failed_without_receiver=not agents and parsed.outer_status == "failed",
        call_id=parsed.call_id,
        reopens_transient=(
            parsed.action in CHILD_GENERATION_ACTIONS or parsed.reopens_transient
        ),
        malformed=parsed.malformed,
        container_malformed=parsed.container_malformed,
    )


def project_activity(acp_event: Any) -> ActivityProjection | None:
    tool_call = getattr(acp_event, "tool_call", None)
    source_id = strict_source_id(
        getattr(tool_call, "subagent_source_id", None)
        or getattr(acp_event, "source_id", None)
    )
    raw_activity = getattr(tool_call, "subagent_activity", None)
    activity = (
        raw_activity.strip().casefold()
        if isinstance(raw_activity, str)
        else ""
    )
    if tool_call is None or not source_id or not activity:
        return None
    if activity not in KNOWN_CHILD_ACTIVITIES:
        return ActivityProjection(
            source_id=source_id,
            label=extract_agent_task_label(tool_call),
            status="malformed",
            progress="状态未知",
            interrupted=False,
            malformed=True,
            reopens=False,
        )
    provider_status = str(getattr(tool_call, "status", "") or "").strip().casefold()
    running, completed, failed = _ACTIVITY_PROGRESS[activity]
    progress = (
        completed
        if provider_status == "completed"
        else failed
        if provider_status == "failed"
        else running
    )
    interrupted = activity == "interrupted" and provider_status == "completed"
    reopens = activity in {"started", "interacted"} or (
        activity == "interrupted" and not interrupted
    )
    return ActivityProjection(
        source_id=source_id,
        label=extract_agent_task_label(tool_call),
        status="cancelled" if interrupted else "running",
        progress=progress,
        interrupted=interrupted,
        malformed=False,
        reopens=reopens,
    )


def _next_sequence(summaries: Mapping[str, Mapping], parent_sequence: object) -> str:
    return f"{parent_sequence}.{chr(ord('a') + len(summaries))}"


def _entry_from_summary(
    source_id: str,
    existing: Mapping[str, Any],
) -> ChildLifecycleEntry | None:
    if not existing:
        return None
    raw_status = str(existing.get("status") or "").strip().casefold()
    malformed = bool(existing.get("malformed")) or (
        bool(raw_status)
        and raw_status not in TERMINAL_CHILD_STATUSES
        and raw_status not in TRANSIENT_CHILD_STATUSES
    )
    status = None if malformed or not raw_status else raw_status
    return ChildLifecycleEntry(
        source_id=source_id,
        status=status,
        raw_status=raw_status,
        generation=max(1, int(existing.get("generation") or 1)),
        provenance=ChildObservationProvenance.COLLABORATION_SNAPSHOT,
        malformed=malformed,
        active_call_id="",
    )


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
        prior_entry = _entry_from_summary(agent.source_id, existing)
        prior_status = (
            prior_entry.status if prior_entry is not None else ""
        )
        incoming_malformed = agent.malformed or projection.container_malformed
        incoming_status = (
            None
            if incoming_malformed or agent.status == "malformed"
            else agent.status
        )
        if (
            prior_entry is not None
            and not prior_entry.malformed
            and prior_status in TERMINAL_AGENT_STATUSES
            and not projection.starts_new_generation
        ):
            if incoming_status != prior_status and not (
                projection.authoritative_list_snapshot
                and incoming_status in TERMINAL_AGENT_STATUSES
            ):
                if incoming_status is not None:
                    continue
            elif (
                incoming_status == prior_status
                and (
                    not agent.progress
                    or agent.progress == str(existing.get("progress") or "")
                )
            ):
                continue

        reopen = projection.starts_new_generation or (
            projection.reopens_transient
            and incoming_status in TRANSIENT_CHILD_STATUSES
        )
        next_entry = reduce_child_entry(
            prior_entry,
            source_id=agent.source_id,
            status=incoming_status,
            raw_status=agent.raw_status,
            provenance=agent.provenance,
            malformed=incoming_malformed,
            reopen=reopen,
            reopen_while_transient=projection.reopens_transient,
            call_id=projection.call_id,
            message=agent.progress,
        )
        next_status = (
            "malformed"
            if next_entry.malformed or next_entry.status is None
            else next_entry.status
        )
        progress = (
            "状态未知"
            if next_status == "malformed"
            else agent.progress
        )
        label = str(existing.get("label") or agent.label).strip()
        if is_generic_task_label(label):
            label = "子任务"
        summary = {
            **existing,
            "label": label,
            "tool": str(existing.get("tool") or "子代理"),
            "status": next_status,
            "progress": progress,
            "generation": next_entry.generation,
        }
        if next_entry.malformed:
            summary["malformed"] = True
        if next_status != "failed":
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
    prior_entry = _entry_from_summary(projection.source_id, existing)
    incoming_status = (
        None
        if projection.malformed or projection.status == "malformed"
        else projection.status
    )
    if (
        prior_entry is not None
        and not prior_entry.malformed
        and prior_entry.status in TERMINAL_AGENT_STATUSES
        and not projection.reopens
        and not projection.interrupted
        and not projection.malformed
    ):
        return dict(summaries)

    next_entry = reduce_child_entry(
        prior_entry,
        source_id=projection.source_id,
        status=incoming_status,
        raw_status=projection.status,
        provenance=ChildObservationProvenance.SUBAGENT_ACTIVITY,
        malformed=projection.malformed,
        reopen=projection.reopens,
        message=projection.progress,
    )
    next_status = (
        "malformed"
        if next_entry.malformed or next_entry.status is None
        else next_entry.status
    )
    progress = "状态未知" if next_status == "malformed" else projection.progress
    label = str(existing.get("label") or projection.label).strip()
    if is_generic_task_label(label):
        label = "子任务"
    summary = {
        **existing,
        "label": label,
        "tool": str(existing.get("tool") or "子代理"),
        "status": next_status,
        "progress": progress,
        "generation": next_entry.generation,
    }
    if next_entry.malformed:
        summary["malformed"] = True
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
        if not existing.get("malformed") and status in TERMINAL_AGENT_STATUSES:
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
