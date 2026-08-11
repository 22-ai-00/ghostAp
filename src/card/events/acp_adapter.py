"""ACP event to CardEvent adaptation logic.

Extracted from CardEvent.from_acp() to maintain SRP — the CardEvent class
stays focused on being a pure data container with simple factory methods,
while this module handles the ACP protocol translation concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.card.tool_display import (
    collect_subagent_opaque_values,
    sanitize_full_tool_event_content,
    sanitize_tool_event_content,
    sanitize_tool_failure_detail,
)

from .factories import CardEvent
from .types import CardEventType

if TYPE_CHECKING:
    from src.acp.models import ACPEvent


def _full_tool_payload(tool_call: object | None) -> str:
    if tool_call is None:
        return ""
    opaque_ids = collect_subagent_opaque_values(tool_call)
    full_content = getattr(tool_call, "full_content", None)
    content = sanitize_full_tool_event_content(
        (
            full_content
            if full_content is not None
            else getattr(tool_call, "content", "")
        ),
        opaque_ids=opaque_ids,
    )
    result = getattr(tool_call, "result", None)
    if result is None:
        return content
    structured = sanitize_full_tool_event_content(result, opaque_ids=opaque_ids)
    if not structured or structured in content:
        return content
    if not content:
        return structured
    return f"{content}\n\n结构化结果\n{structured}"


def card_event_from_acp(
    acp_event: "ACPEvent",
    *,
    preserve_tool_content: bool = False,
) -> CardEvent:
    """Convert an ACPEvent to a CardEvent.

    Maps ACP event types to the card event pipeline:
    - TEXT_CHUNK → TEXT_DELTA
    - THOUGHT_CHUNK → REASONING_DELTA
    - IMAGE_CHUNK → IMAGE_FAILED unless the media bridge uploads it first
    - TOOL_CALL_START → TOOL_STARTED
    - TOOL_CALL_UPDATE → TOOL_DELTA
    - TOOL_CALL_DONE → TOOL_DONE / TOOL_FAILED
    - PLAN_UPDATE → TASK_LIST_UPDATED
    - (fallback) → TEXT_DELTA
    """
    from src.acp.models import ACPEventType as AET

    match acp_event.event_type:
        case AET.TEXT_CHUNK:
            return CardEvent(type=CardEventType.TEXT_DELTA, payload={
                "block_id": "_active_text",
                "text": acp_event.text or "",
            })
        case AET.THOUGHT_CHUNK:
            return CardEvent(type=CardEventType.REASONING_DELTA, payload={
                "block_id": "_active_reasoning",
                "text": acp_event.text or "",
            })
        case AET.IMAGE_CHUNK:
            image = acp_event.image
            return CardEvent.image_failed(
                image.image_id if image else "",
                image.name if image else "任务图片",
            )
        case AET.TOOL_CALL_START:
            tc = acp_event.tool_call
            content = (
                _full_tool_payload(tc)
                if preserve_tool_content
                else sanitize_tool_event_content(
                    tc.content if tc else "",
                    fallback=tc.title if tc else "",
                )
            )
            return CardEvent(type=CardEventType.TOOL_STARTED, payload={
                "block_id": tc.id if tc else "",
                "tool_name": tc.title if tc else "",
                "tool_input": content,
            })
        case AET.TOOL_CALL_UPDATE:
            tc = acp_event.tool_call
            content = (
                _full_tool_payload(tc)
                if preserve_tool_content
                else sanitize_tool_event_content(
                    tc.content if tc else "",
                    fallback=tc.title if tc else "",
                )
            )
            return CardEvent(type=CardEventType.TOOL_DELTA, payload={
                "block_id": tc.id if tc else "",
                "tool_name": tc.title if tc else "",
                "content": content,
            })
        case AET.TOOL_CALL_DONE:
            tc = acp_event.tool_call
            summary = tc.title if tc else ""
            output = (
                _full_tool_payload(tc)
                if preserve_tool_content
                else sanitize_tool_event_content(
                    tc.content if tc else "",
                    fallback=summary,
                )
            )
            status = tc.status if tc else "completed"
            if status == "failed":
                if preserve_tool_content:
                    output = _full_tool_payload(tc)
                else:
                    output = sanitize_tool_failure_detail(
                        tc.content if tc else "",
                        fallback=summary or "工具执行失败",
                        opaque_ids=(tc.id,) if tc else (),
                    )
                if not output:
                    output = summary or "工具执行失败"
                return CardEvent(type=CardEventType.TOOL_FAILED, payload={
                    "block_id": tc.id if tc else "",
                    "tool_name": summary,
                    "error": output,
                })
            return CardEvent(type=CardEventType.TOOL_DONE, payload={
                "block_id": tc.id if tc else "",
                "tool_output": output,
                "tool_summary": summary,
            })
        case AET.PLAN_UPDATE:
            plan = acp_event.plan
            tasks = []
            current_task_id = ""
            if plan:
                from src.card.task_registry import tasks_from_plan_entries

                tasks = tasks_from_plan_entries(plan.entries)
                current_task_id = next(
                    (
                        str(task.get("task_id") or "")
                        for task in tasks
                        if task.get("status") == "in_progress"
                    ),
                    "",
                )
            return CardEvent(
                type=CardEventType.TASK_LIST_UPDATED,
                payload={"tasks": tasks, "current_task_id": current_task_id},
            )
        case _:
            return CardEvent(type=CardEventType.TEXT_DELTA, payload={
                "block_id": "_active_text",
                "text": acp_event.text or "",
            })
