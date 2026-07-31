"""Slock Engine card templates — LEGACY monolithic module (pending migration).

⚠️ MIGRATION STATUS:
Functions ALREADY migrated to card_templates/ subpackage (do NOT modify here):
  - build_status_panel_card → card_templates/status.py
  - build_role_info_card → card_templates/role.py
  - build_role_list_card → card_templates/role.py
  - build_task_board_card → card_templates/task.py
  - build_progress_overview_card → card_templates/progress.py
  - build_collaboration_plan_card → card_templates/progress.py
  - build_discussion_live_card → card_templates/discussion.py
  - build_discussion_conclusion_card → card_templates/discussion.py
  - build_discussion_history_list_card → card_templates/discussion.py

Functions PENDING migration (still authoritative in this file):
  - build_welcome_card
  - build_team_created_card
  - build_team_list_card
  - build_command_hub_card
  - build_command_panel_card / build_command_panel_extended_card
  - build_council_card / build_council_detail_card / build_council_result_card
  - build_discussion_card / build_discussion_history_card / build_discussion_summary_card
  - build_memory_display_card / build_memory_manage_card
  - build_role_switch_card / build_role_arg_error_card
  - build_agent_message_card / build_agent_action_buttons
  - build_escalation_card / build_conflict_escalation_card / build_resolved_escalation_card
  - build_dissolve_confirm_card / build_dissolve_undo_card
  - build_crash_recovery_card / build_error_suggestion_card
  - build_status_refresh_card
  - All other build_* and helper functions below

This file will be deleted once all functions are migrated.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

from src.card.shared import apply_compact_style, build_responsive_layout
from src.utils.redact import redact_sensitive

from .models import (
    ABORT_OPTIONS,
    AGENT_STATUS_BG_COLOR_MAP,
    AgentIdentity,
    AgentStatus,
    CouncilRun,
    CouncilStatus,
    EscalationLevel,
    EscalationRequest,
    SlockMemory,
    SlockTask,
    TaskStatus,
)

_DISPLAY_TZ = ZoneInfo("Asia/Shanghai")


# ---------------------------------------------------------------------------
# Local card wrapper — mirrors card_templates/common.py:build_card_wrapper
# to ensure visual consistency without circular imports.
# ---------------------------------------------------------------------------

_CARD_BYTE_BUDGET = 27 * 1024
_CARD_NODE_BUDGET = 180


def _count_tagged_nodes(obj) -> int:
    """Count dicts with a 'tag' key (Feishu element nodes)."""
    count = 0
    if isinstance(obj, dict):
        if "tag" in obj:
            count += 1
        for v in obj.values():
            count += _count_tagged_nodes(v)
    elif isinstance(obj, list):
        for item in obj:
            count += _count_tagged_nodes(item)
    return count


def _guard_card_payload(card: dict) -> dict:
    """Truncate elements from the end if card exceeds byte/node budget."""
    import json

    elements = card.get("body", {}).get("elements", [])
    if not elements:
        return card

    serialized = json.dumps(card, ensure_ascii=False)
    byte_size = len(serialized.encode("utf-8"))
    node_count = _count_tagged_nodes(card)

    if byte_size <= _CARD_BYTE_BUDGET and node_count <= _CARD_NODE_BUDGET:
        return card

    while len(elements) > 1 and (byte_size > _CARD_BYTE_BUDGET or node_count > _CARD_NODE_BUDGET):
        elements.pop()
        card["body"]["elements"] = elements
        serialized = json.dumps(card, ensure_ascii=False)
        byte_size = len(serialized.encode("utf-8"))
        node_count = _count_tagged_nodes(card)

    elements.append({"tag": "markdown", "content": "*⚠️ 内容过长，部分已截断*"})
    card["body"]["elements"] = elements
    return card


def _build_card_wrapper(
    *,
    header_title: str,
    header_template: str = "indigo",
    header_subtitle: str = "",
    elements: list[dict],
    mobile_optimize: bool = True,
) -> dict:
    """Wrap elements into a Feishu Interactive Card 2.0 structure with payload guard.

    Mirrors card_templates/common.py:build_card_wrapper for visual consistency.
    """
    header: dict = {
        "title": {"tag": "plain_text", "content": header_title},
        "template": header_template,
    }
    if header_subtitle:
        header["subtitle"] = {"tag": "plain_text", "content": header_subtitle}

    card = {
        "schema": "2.0",
        "config": {"wide_screen_mode": not mobile_optimize},
        "header": header,
        "body": {"elements": elements},
    }
    return _guard_card_payload(card)

_STATUS_LABEL_ZH: dict[str, str] = {
    "idle": "空闲",
    "waking": "唤醒中",
    "thinking": "思考中",
    "running": "运行中",
    "checking": "检查中",
    "sending": "发送中",
    "moving": "迁移中",
    "discussing": "讨论中",
    "pending_discussion": "等待确认",
}

_TASK_STATUS_LABEL_ZH: dict[str, str] = {
    "todo": "待办",
    "in_progress": "进行中",
    "in_review": "审查中",
    "done": "已完成",
}


def _make_collapsible(title: str, elements: list[dict]) -> dict:
    """Create a Feishu collapsible_panel element (collapsed by default)."""
    return {
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": title}},
        "vertical_spacing": "8px",
        "elements": elements,
    }


def build_agent_message_card(
    agent: AgentIdentity,
    content: str,
    *,
    model_info: str = "",
    duration_s: Optional[float] = None,
    channel_id: str = "",
    task_id: str = "",
    discussion_enabled: bool = True,
) -> dict:
    """Build an Interactive Card for an agent's message (mouthpiece output).

    Args:
        agent: The agent identity sending the message.
        content: Markdown content of the message.
        model_info: Optional model identifier for footer.
        duration_s: Optional processing duration in seconds.
    """
    header_title = agent.display_name
    header_template = agent.card_color

    elements: list[dict] = []

    # Main content — redact sensitive tokens before rendering
    content = redact_sensitive(content)
    elements.append({"tag": "markdown", "content": content})

    # Footer note with metadata
    footer_parts: list[str] = []
    if agent.agent_type:
        footer_parts.append(agent.agent_type)
    if model_info:
        footer_parts.append(model_info)
    elif agent.model_name:
        footer_parts.append(agent.model_name)
    if duration_s is not None:
        if duration_s < 60:
            footer_parts.append(f"{duration_s:.1f}s")
        else:
            minutes = int(duration_s // 60)
            secs = duration_s % 60
            footer_parts.append(f"{minutes}m{secs:.0f}s")

    if footer_parts:
        elements.append({
            "tag": "markdown",
            "content": " | ".join(footer_parts),
            "text_size": "notation",
        })

    action_value = {
        "channel_id": channel_id,
        "agent_id": agent.agent_id,
        "agent_name": agent.name,
        "task_id": task_id,
    }

    # Row 1: core action buttons (3 buttons: ask, done, continue)
    elements.extend(
        build_responsive_layout(
            [
                _build_callback_button(
                    "@追问",
                    "slock_agent_follow_up",
                    channel_id=channel_id,
                    button_type="primary",
                    extra_value=action_value,
                ),
                _build_callback_button(
                    "标记完成",
                    "slock_agent_mark_done",
                    channel_id=channel_id,
                    button_type="default",
                    extra_value=action_value,
                ),
                _build_callback_button(
                    "▶️ 让TA继续",
                    "slock_agent_continue",
                    channel_id=channel_id,
                    button_type="default",
                    extra_value=action_value,
                ),
            ]
        )
    )

    # Row 2: secondary buttons in collapsible panel (AC23: mobile-friendly)
    secondary_buttons = [
        _build_callback_button(
            "查看推理",
            "slock_agent_show_reasoning",
            channel_id=channel_id,
            button_type="text",
            extra_value=action_value,
        ),
        _build_callback_button(
            "🧠 查看记忆",
            "slock_agent_show_memory",
            channel_id=channel_id,
            button_type="text",
            extra_value=action_value,
        ),
        _build_callback_button(
            "换个角色",
            "slock_agent_switch_role",
            channel_id=channel_id,
            button_type="text",
            extra_value=action_value,
        ),
    ]
    if discussion_enabled:
        secondary_buttons.append(
            _build_callback_button(
                "开启讨论",
                "slock_start_discussion",
                channel_id=channel_id,
                button_type="text",
                extra_value=action_value,
            )
        )
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {
            "title": {
                "tag": "markdown",
                "content": "**⚙️ 更多（讨论/记忆/换角色）**",
            },
        },
        "vertical_spacing": "8px",
        "elements": build_responsive_layout(secondary_buttons),
    })

    return _build_card_wrapper(
        header_title=header_title,
        header_template=header_template,
        elements=elements,
        mobile_optimize=True,
    )


def build_agent_action_buttons(
    *,
    channel_id: str = "",
    extra_value: dict | None = None,
) -> list[dict]:
    """Create inline action buttons for the agent response card.

    Returns a list of button dicts suitable for build_responsive_layout().

    Buttons:
        - "让TA继续" — action: slock_agent_continue
        - "换个角色" — action: slock_agent_switch_role
        - "开启讨论" — action: slock_start_discussion
    """
    base_value = extra_value or {}
    return [
        _build_callback_button(
            "让TA继续",
            "slock_agent_continue",
            channel_id=channel_id,
            extra_value=base_value,
        ),
        _build_callback_button(
            "换个角色",
            "slock_agent_switch_role",
            channel_id=channel_id,
            extra_value=base_value,
        ),
        _build_callback_button(
            "开启讨论",
            "slock_start_discussion",
            channel_id=channel_id,
            extra_value=base_value,
        ),
    ]


def build_nli_feedback_card(
    intent_description: str,
    channel_id: str,
    intent_params: dict,
) -> dict:
    """Build a confirmation card shown after NLI intent recognition.

    Displays what intent was recognized and offers confirm/cancel buttons.

    Args:
        intent_description: Human-readable description of the recognized intent
            (e.g., "创建一个新角色").
        channel_id: Channel ID for action routing.
        intent_params: Dict of parsed intent parameters to pass back on confirm.
    """
    elements: list[dict] = []

    elements.append({
        "tag": "markdown",
        "content": f"我理解你想：**{intent_description}**",
    })

    action_value = {
        "channel_id": channel_id,
        "intent_params": intent_params,
    }

    elements.extend(
        build_responsive_layout(
            [
                _build_callback_button(
                    "确认执行",
                    "slock_nli_confirm",
                    channel_id=channel_id,
                    button_type="primary",
                    extra_value=action_value,
                ),
                _build_callback_button(
                    "取消",
                    "slock_nli_cancel",
                    channel_id=channel_id,
                    extra_value=action_value,
                ),
            ]
        )
    )

    return _build_card_wrapper(
        header_title="🤔 意图识别",
        header_template="wathet",
        elements=elements,
        mobile_optimize=False,
    )


def build_team_created_card(
    *,
    team_name: str,
    group_name: str,
    channel_id: str,
) -> dict:
    """Build the /new-team confirmation card with a direct group jump button."""
    content = (
        f"✅ 团队 **{team_name}** 已创建\n\n"
        f"已建立专属协作群「{group_name}」并激活 Slock 运行时\n"
        "• 事件监听: ✓\n"
        "• Agent 调度器: ✓\n"
        "• 工作区目录: ✓\n\n"
        "请前往新群开始协作"
    )
    elements: list[dict] = [{"tag": "markdown", "content": content}]
    elements.extend(build_responsive_layout([_build_slock_group_jump_button(channel_id)]))
    return _build_card_wrapper(
        header_title="🎭 Slock 协作群已创建",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )


def build_welcome_card(*, team_name: str) -> dict:
    """Build a welcome card sent inside the newly created Slock team group."""
    content = (
        f"🎭 **协作团队「{team_name}」已就绪**\n\n"
        "团队已开启自主工作模式，员工 Agent 会自动响应群内消息。\n\n"
        "**快速开始:**\n"
        "• `/hire <名字>` — 雇佣新员工（选工具+模型）\n"
        "• `/role add` — 添加已有员工到本群\n"
        "• `/role list` — 查看群内员工\n"
        "• 直接发送任务描述 — 创建团队任务\n"
        "• 直接 @员工名 发消息 — 指定员工执行任务\n\n"
        "**管理命令:**\n"
        "`/role remove <名字>` · `/task status` · `/slock status` · `/team dissolve`"
    )
    elements: list[dict] = [{"tag": "markdown", "content": content}]
    return _build_card_wrapper(
        header_title=f"👋 欢迎加入 {team_name}",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )


def build_team_list_card(teams: list[dict]) -> dict:
    """Build a Slock team directory card with jump buttons."""
    elements: list[dict] = []
    for team in teams:
        team_name = str(team.get("team_name") or team.get("name") or team.get("channel_id") or "")
        channel_id = str(team.get("channel_id") or "")
        agent_count = int(team.get("agent_count") or 0)
        task_count = int(team.get("task_count") or 0)
        elements.append(
            {
                "tag": "markdown",
                "content": (
                    f"**{team_name}**\n"
                    f"{agent_count} 个 Agent · {task_count} 个任务 · 频道: `{channel_id}`"
                ),
            }
        )
        if channel_id:
            elements.extend(build_responsive_layout([_build_slock_group_jump_button(channel_id)]))
        elements.append({"tag": "hr"})

    if elements and elements[-1].get("tag") == "hr":
        elements.pop()

    return _build_card_wrapper(
        header_title="🎭 Slock 协作群",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )



# ------------------------------------------------------------------
# Agent move notification
# ------------------------------------------------------------------


def build_agent_move_departure_card(
    agent: AgentIdentity,
    target_team: str,
) -> dict:
    """Build a departure notification card sent to the SOURCE group.

    Informs all members that an agent has been moved out.  Uses orange header
    to signal a change event (not green/completion).  No jump button — keeps
    the card simple and informational.

    Args:
        agent: The agent identity being moved away.
        target_team: Display name of the target team/group.
    """
    from datetime import datetime

    content = (
        f"{agent.emoji} **{agent.name}** 已迁出至「{target_team}」\n\n"
        "角色记忆与技能画像已随迁，如需协作请前往目标团队。"
    )

    elements: list[dict] = [{"tag": "markdown", "content": content}]

    # Footer note
    footer_parts: list[str] = []
    if agent.agent_type:
        footer_parts.append(agent.agent_type)
    if agent.model_name:
        footer_parts.append(agent.model_name)
    footer_parts.append(datetime.now().strftime("%Y-%m-%d %H:%M"))
    elements.append({
        "tag": "markdown",
        "content": " | ".join(footer_parts),
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title=f"➡️ 角色迁出: {agent.name}",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )


def build_agent_move_notification_card(
    agent: AgentIdentity,
    source_team: str,
    target_team: str,
    *,
    operator_display: str = "",
) -> dict:
    """Build a notification card for an agent being moved to a new team.

    Sent to the TARGET group to inform members about the newly arrived agent.
    Does not include a jump button (the target group IS the destination).

    Args:
        agent: The agent identity being moved.
        source_team: Display name of the source team/group.
        target_team: Display name of the target team/group.
        operator_display: Display name of the operator who initiated the move.
    """
    operator_line = f"由 {operator_display} 迁移至此" if operator_display else "已迁移至此"
    content = (
        f"{agent.emoji} **{agent.name}** 已从「{source_team}」迁移至本团队\n\n"
        f"• 角色: {agent.role or 'custom'}\n"
        f"• 工具: {agent.agent_type}\n"
        f"• 模型: {agent.model_name or '默认'}\n"
        f"• {operator_line}\n\n"
        "角色定义、关键知识与技能画像已保留；活跃上下文已按跨群策略重置，可立即参与任务分配。"
    )

    elements: list[dict] = [{"tag": "markdown", "content": content}]

    # Footer note: agent_type | model_name | timestamp
    from datetime import datetime

    footer_parts: list[str] = []
    if agent.agent_type:
        footer_parts.append(agent.agent_type)
    if agent.model_name:
        footer_parts.append(agent.model_name)
    footer_parts.append(datetime.now().strftime("%Y-%m-%d %H:%M"))
    elements.append({
        "tag": "markdown",
        "content": " | ".join(footer_parts),
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title=f"👋 角色加入: {agent.name}",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )


def build_agent_move_confirm_card(
    agent: AgentIdentity,
    source_team: str,
    target_team: str,
    target_channel_id: str,
) -> dict:
    """Build a confirmation card sent to the SOURCE group after a successful move.

    Includes a jump button pointing to the target group so the operator can
    quickly follow the agent to its new home.

    Args:
        agent: The agent identity that was moved.
        source_team: Display name of the source team/group.
        target_team: Display name of the target team/group.
        target_channel_id: Channel ID of the target group for jump button.
    """
    content = (
        f"✅ 角色 **{agent.emoji} {agent.name}** 已成功移动到团队「{target_team}」\n\n"
        f"• 角色: {agent.role or 'custom'}\n"
        f"• 工具: {agent.agent_type}\n"
        f"• 模型: {agent.model_name or '默认'}\n\n"
        "角色定义与技能画像已保留，活跃上下文已按跨群隐私策略脱敏。\n\n"
        "ℹ️ Active Context 已按跨群策略重置以保护源团队隐私"
    )

    elements: list[dict] = [{"tag": "markdown", "content": content}]

    if target_channel_id:
        elements.extend(
            build_responsive_layout([_build_slock_group_jump_button(target_channel_id)])
        )

    # Footer note: agent_type | model_name | timestamp
    from datetime import datetime

    footer_parts: list[str] = []
    if agent.agent_type:
        footer_parts.append(agent.agent_type)
    if agent.model_name:
        footer_parts.append(agent.model_name)
    footer_parts.append(datetime.now().strftime("%Y-%m-%d %H:%M"))
    elements.append({
        "tag": "markdown",
        "content": " | ".join(footer_parts),
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title=f"✅ 角色迁移完成: {agent.name}",
        header_template="green",
        elements=elements,
        mobile_optimize=False,
    )


# ------------------------------------------------------------------
# Discussion cards
# ------------------------------------------------------------------


def build_discussion_card_from_thread(thread, engine=None) -> dict:
    """Factory: build discussion card directly from a DiscussionThread object.

    Extracts fields from the thread dataclass and delegates to
    build_discussion_card (keyword-only signature).
    """
    # Resolve participant IDs to display names
    participants_display = thread.participants
    if engine:
        registry = getattr(engine, 'registry', None) or getattr(engine, '_registry', None)
        if registry:
            resolved = []
            for pid in thread.participants:
                agent = registry.get_agent(pid) if hasattr(registry, 'get_agent') else None
                if agent:
                    display = f"{getattr(agent, 'emoji', '')} {agent.name}".strip()
                    resolved.append(display)
                else:
                    resolved.append(pid)
            participants_display = resolved

    # Resolve message sender IDs and redact sensitive content
    messages = []
    for m in thread.messages:
        msg_dict = m.to_dict()
        # AC16: Redact sensitive info (API keys, tokens) before rendering to card
        if 'content' in msg_dict:
            msg_dict['content'] = redact_sensitive(msg_dict['content'])
        if engine:
            registry = getattr(engine, 'registry', None) or getattr(engine, '_registry', None)
            if registry:
                agent = registry.get_agent(m.sender_agent_id) if hasattr(registry, 'get_agent') else None
                if agent:
                    msg_dict['sender'] = f"{getattr(agent, 'emoji', '')} {agent.name}".strip()
                    msg_dict['sender_display_name'] = msg_dict['sender']
        messages.append(msg_dict)

    return build_discussion_card(
        thread_id=thread.thread_id,
        participants=participants_display,
        messages=messages,
        current_round=thread.current_round,
        max_rounds=thread.config.max_rounds,
        trigger_reason=thread.trigger_reason,
        channel_id=thread.channel_id,
    )


def build_discussion_summary_card_from_thread(thread, engine=None) -> dict:
    """Factory: build discussion summary card from a completed DiscussionThread.

    Extracts fields from the thread dataclass and delegates to
    build_discussion_summary_card (keyword-only signature).
    """
    # Resolve participant IDs to display names
    participants_display = thread.participants
    if engine:
        registry = getattr(engine, 'registry', None) or getattr(engine, '_registry', None)
        if registry:
            resolved = []
            for pid in thread.participants:
                agent = registry.get_agent(pid) if hasattr(registry, 'get_agent') else None
                if agent:
                    display = f"{getattr(agent, 'emoji', '')} {agent.name}".strip()
                    resolved.append(display)
                else:
                    resolved.append(pid)
            participants_display = resolved

    # AC16: Redact sensitive info from conclusion before rendering
    conclusion = redact_sensitive(thread.conclusion) if thread.conclusion else ""

    return build_discussion_summary_card(
        thread_id=thread.thread_id,
        participants=participants_display,
        conclusion=conclusion,
        total_rounds=thread.current_round,
        total_tokens=thread.total_tokens_used,
        status=thread.status.value if hasattr(thread.status, "value") else str(thread.status),
        channel_id=thread.channel_id,
    )


def build_discussion_card(
    *,
    thread_id: str,
    participants: list[str],
    messages: list[dict],
    current_round: int,
    max_rounds: int,
    trigger_reason: str = "",
    channel_id: str = "",
) -> dict:
    """Build a live discussion thread card showing agent-to-agent dialogue.

    Args:
        thread_id: Unique discussion thread identifier.
        participants: List of agent display names in the discussion.
        messages: List of dicts with keys: sender, content, round_num.
        current_round: Current round number.
        max_rounds: Maximum allowed rounds.
        trigger_reason: Why the discussion was triggered.
        channel_id: Channel ID for action buttons.
    """
    header_title = f"💬 讨论 R{current_round}/{max_rounds}"
    if len(participants) > 2:
        header_subtitle = f"👥 {' · '.join(participants[:2])} 等 {len(participants)} 人"
    else:
        header_subtitle = f"👥 {' · '.join(participants)}"

    elements: list[dict] = []

    # Participants line
    elements.append({
        "tag": "markdown",
        "content": f"**参与者:** {' ↔ '.join(participants)}",
    })

    # Progress bar: single-line Markdown for mobile compatibility (AC24: ≤30 chars)
    if max_rounds >= 10:
        # Percentage + fraction format for large round counts
        pct = round(current_round / max_rounds * 100)
        progress_text = f"**进度:** {pct}% ({current_round}/{max_rounds})"
    else:
        progress_filled = "●" * current_round
        progress_empty = "○" * (max_rounds - current_round)
        progress_text = f"**进度:** {progress_filled}{progress_empty} ({current_round}/{max_rounds})"
    elements.append({
        "tag": "markdown",
        "content": progress_text,
    })

    if trigger_reason:
        # Determine colored tag based on trigger source
        if "manual" in trigger_reason:
            _trigger_color = "blue"
            _trigger_label = "人工触发"
        elif "uncertainty" in trigger_reason or "auto_uncertainty" in trigger_reason:
            _trigger_color = "orange"
            _trigger_label = "系统检测"
        elif "chain" in trigger_reason or "auto_chain" in trigger_reason:
            _trigger_color = "green"
            _trigger_label = "链式触发"
        else:
            _trigger_color = "neutral"
            _trigger_label = trigger_reason
        elements.append({
            "tag": "markdown",
            "content": f"**触发原因:** <font color='{_trigger_color}'>{_trigger_label}</font>",
            "text_size": "notation",
        })

    elements.append({"tag": "hr"})

    # Show last N messages (limit to 8 to keep card compact)
    display_messages = messages[-8:] if len(messages) > 8 else messages
    if len(messages) > 8:
        elements.append({
            "tag": "markdown",
            "content": f"*... 已省略 {len(messages) - 8} 条早期消息*",
            "text_size": "notation",
        })

    _CONTENT_THRESHOLD = 120  # Threshold for using collapsible_panel

    for msg in display_messages:
        sender = msg.get("sender", "Agent")
        raw_content = msg.get("content", "")
        round_num = msg.get("round_num", "?")

        if len(raw_content) > _CONTENT_THRESHOLD:
            # Long message: wrap in collapsible panel for mobile-friendly display
            md_element = {"tag": "markdown", "content": f"💬 **{sender}** (R{round_num}):\n{raw_content}"}
            # Header shows truncated preview (first 120 chars)
            preview = raw_content[:_CONTENT_THRESHOLD] + "…" if len(raw_content) > _CONTENT_THRESHOLD else raw_content
            elements.append(
                _make_collapsible(f"💬 {sender} (R{round_num}): {preview}", [md_element])
            )
        else:
            # Short message: use note for compact mobile display.
            elements.append({
                "tag": "note",
                "icon": {"tag": "standard_icon", "token": "chat_outlined"},
                "elements": [
                    {"tag": "markdown", "content": f"**{sender}** (R{round_num}): {raw_content}"},
                ],
            })

    # Action buttons: expand / inject hint / stop discussion (3 buttons for mobile horizontal layout)
    elements.append({"tag": "hr"})
    elements.extend(
        build_responsive_layout(
            [
                _build_callback_button(
                    "📖 展开全部",
                    "slock_discussion_expand",
                    channel_id=channel_id,
                    extra_value={"thread_id": thread_id},
                ),
                _build_callback_button(
                    "💡 人工干预",
                    "inject_discussion_hint",
                    channel_id=channel_id,
                    button_type="primary",
                    extra_value={"thread_id": thread_id},
                ),
                _build_callback_button(
                    "⏹ 停止讨论",
                    "slock_discussion_stop",
                    channel_id=channel_id,
                    button_type="danger",
                    extra_value={"thread_id": thread_id},
                ),
            ]
        )
    )

    return _build_card_wrapper(
        header_title=header_title,
        header_template="purple",
        header_subtitle=header_subtitle,
        elements=elements,
        mobile_optimize=False,
    )


def build_discussion_expand_card(
    *,
    thread_id: str,
    messages: list[dict],
    participants: list[str],
    channel_id: str = "",
    page: int = 0,
) -> dict:
    """Build a paginated discussion thread card (true pagination)."""
    PAGE_SIZE = 10
    elements: list[dict] = []

    # Header participants line
    if len(participants) > 2:
        participants_text = f"👥 {' · '.join(participants[:2])} 等 {len(participants)} 人"
    else:
        participants_text = f"👥 {' · '.join(participants)}"
    elements.append({"tag": "markdown", "content": participants_text})

    # Paginate: only render current page
    total = len(messages)
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page_messages = messages[start:end]

    elements.append({
        "tag": "markdown",
        "content": f"第 {start + 1}-{end} 条，共 {total} 条",
        "text_size": "notation",
    })

    for msg in page_messages:
        sender = msg.get("sender", "Agent")
        content = msg.get("content", "")
        round_num = msg.get("round_num", "?")
        elements.append({
            "tag": "note",
            "elements": [
                {
                    "tag": "standard_icon",
                    "token": "chat_outlined",
                    "color": "grey",
                },
                {
                    "tag": "plain_text",
                    "content": f"{sender} (R{round_num}): {content}",
                },
            ],
        })

    # "Load more" button only if there are more messages
    if end < total:
        next_page = page + 1
        elements.append({"tag": "hr"})
        elements.extend(
            build_responsive_layout(
                [
                    _build_callback_button(
                        f"加载更多 ({end}/{total})",
                        "slock_discussion_expand_page",
                        channel_id=channel_id,
                        extra_value={
                            "thread_id": thread_id,
                            "page_num": next_page,
                        },
                    ),
                ]
            )
        )

    return _build_card_wrapper(
        header_title=f"💬 讨论详情 — {thread_id[:8]}",
        header_template="purple",
        elements=elements,
        mobile_optimize=False,
    )


def build_discussion_history_card(
    *,
    history: list[dict],
    channel_id: str = "",
) -> dict:
    """Build a card showing discussion history entries from L2 memory.

    Args:
        history: List of dicts with keys: topic_hash, title, participants, time, conclusion.
        channel_id: Channel ID for context.
    """
    elements: list[dict] = []

    for entry in history:
        title = entry.get("title", "Untitled")
        topic_hash = entry.get("topic_hash", "")[:8]
        participants = entry.get("participants", "")
        time_str = entry.get("time", "")
        conclusion = entry.get("conclusion", "")
        if len(conclusion) > 200:
            conclusion = conclusion[:200] + "..."

        entry_text = f"**{title}** `{topic_hash}`\n"
        if participants:
            entry_text += f"👥 {participants}\n"
        if time_str:
            entry_text += f"🕐 {time_str}\n"
        if conclusion:
            entry_text += f"\n{conclusion}"

        elements.append({"tag": "markdown", "content": entry_text})
        elements.append({"tag": "hr"})

    # Remove trailing hr
    if elements and elements[-1].get("tag") == "hr":
        elements.pop()

    if not elements:
        elements.append({"tag": "markdown", "content": "📭 暂无讨论历史。"})

    return _build_card_wrapper(
        header_title="📋 讨论历史",
        header_template="purple",
        elements=elements,
        mobile_optimize=False,
    )


def build_budget_warning_card(
    *,
    thread_id: str,
    current_pct: int,
    channel_id: str = "",
) -> dict:
    """Build a warning card when discussion token budget reaches 80%.

    Args:
        thread_id: The discussion thread identifier.
        current_pct: Current budget usage percentage (e.g. 81).
        channel_id: Channel ID for action buttons.
    """
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": (
                f"⚠️ **讨论预算预警**\n\n"
                f"当前讨论已消耗 **{current_pct}%** 的 Token 预算。\n"
                f"如不干预，预算耗尽后讨论将自动终止。"
            ),
        },
    ]
    elements.extend(
        build_responsive_layout([
            _build_callback_button(
                "⏹ 手动终止",
                "slock_discussion_stop",
                channel_id=channel_id,
                button_type="danger",
                extra_value={"thread_id": thread_id},
            ),
            _build_callback_button(
                "💰 续费（翻倍预算）",
                "slock_discussion_extend_budget",
                channel_id=channel_id,
                button_type="primary",
                extra_value={"thread_id": thread_id},
            ),
        ])
    )

    return _build_card_wrapper(
        header_title="⚠️ 讨论预算预警",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )


def build_discussion_summary_card(
    *,
    thread_id: str,
    participants: list[str],
    conclusion: str,
    total_rounds: int,
    total_tokens: int = 0,
    status: str = "converged",
    channel_id: str = "",
) -> dict:
    """Build a discussion conclusion card after a discussion ends.

    Args:
        thread_id: Unique discussion thread identifier.
        participants: List of agent display names involved.
        conclusion: The final summarized conclusion.
        total_rounds: Total number of rounds completed.
        total_tokens: Total tokens consumed by the discussion.
        status: Final status (converged, timeout, budget_exhausted, manually_stopped).
        channel_id: Channel ID for action buttons.
    """
    status_labels = {
        "converged": ("✅ 达成共识", "green"),
        "timeout": ("⏰ 超时结束", "grey"),
        "max_rounds_reached": ("🔄 已达最大轮次", "grey"),
        "budget_exhausted": ("💰 预算耗尽", "orange"),
        "manually_stopped": ("⏹ 手动终止", "grey"),
    }
    status_display, header_template = status_labels.get(status, (status, "grey"))
    header_title = f"💬 讨论结论 — {status_display}"

    elements: list[dict] = []

    # Participants and stats
    elements.append({
        "tag": "markdown",
        "content": (
            f"**参与者:** {' ↔ '.join(participants)}\n"
            f"**总轮次:** {total_rounds}"
            + (f" · **Token:** {total_tokens:,}" if total_tokens else "")
        ),
    })

    elements.append({"tag": "hr"})

    # Conclusion content
    elements.append({
        "tag": "markdown",
        "content": f"**结论:**\n{conclusion[:500]}",
    })

    # Footer
    elements.append({
        "tag": "markdown",
        "content": f"`thread: {thread_id[:12]}...`",
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title=header_title,
        header_template=header_template,
        elements=elements,
        mobile_optimize=False,
    )


# ------------------------------------------------------------------
# Council cards
# ------------------------------------------------------------------


def build_council_card(run: CouncilRun, *, channel_id: str = "") -> dict:
    """Build a staged Slock Council card."""
    status_label = _COUNCIL_STATUS_LABEL_ZH.get(run.status, run.status.value)
    header_template = "green" if run.status == CouncilStatus.COMPLETED else "red" if run.status == CouncilStatus.FAILED else "indigo"
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": f"**议题:** {redact_sensitive(run.question)[:300]}",
        }
    ]

    if run.error:
        elements.append({"tag": "markdown", "content": f"**错误:** {redact_sensitive(run.error)[:500]}"})

    elements.append({"tag": "hr"})
    elements.append(_build_council_stage_block("1", "独立意见", _format_council_responses(run)))
    elements.append(_build_council_stage_block("2", "匿名互评", _format_council_reviews(run)))
    elements.append(_build_council_stage_block("3", "主席综合", _format_council_final(run)))

    elements.append({
        "tag": "markdown",
        "content": f"`council: {run.run_id[:12]}...` · {status_label}",
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title=f"🧭 Slock Council — {status_label}",
        header_template=header_template,
        elements=elements,
        mobile_optimize=False,
    )


def _build_council_stage_block(index: str, title: str, content: str) -> dict:
    return {
        "tag": "markdown",
        "content": f"**阶段 {index} — {title}**\n{content or '*等待中*'}",
    }


def _format_council_responses(run: CouncilRun) -> str:
    if not run.responses:
        return "*等待 Agent 独立作答*"
    lines: list[str] = []
    for response in run.responses[:6]:
        content = response.content or response.error or "(空)"
        lines.append(
            f"• **{response.label}** · {response.agent_name or response.agent_id[:8]}: "
            f"{redact_sensitive(content)[:240]}"
        )
    return "\n".join(lines)


def _format_council_reviews(run: CouncilRun) -> str:
    if run.aggregate_rankings:
        return "\n".join(
            f"• #{idx + 1} **{item.label}** · {item.agent_name or item.agent_id[:8]} "
            f"(avg {item.average_rank:.2f}, score {item.quality_score:.1f})"
            for idx, item in enumerate(run.aggregate_rankings[:6])
        )
    if run.reviews:
        return "\n".join(
            f"• {review.reviewer_name or review.reviewer_agent_id[:8]}: "
            f"{', '.join(review.parsed_ranking) or '未解析'}"
            for review in run.reviews[:6]
        )
    return "*等待匿名互评*"


def _format_council_final(run: CouncilRun) -> str:
    if run.final_response:
        return redact_sensitive(run.final_response)[:1200]
    return "*等待主席综合*"


def build_council_expandable_card(run: CouncilRun, *, channel_id: str = "") -> dict:
    """Build a compact council card with collapsible panels for each stage.

    Unlike build_council_card which always shows all stages expanded,
    this variant uses collapsible_panel elements — only the latest active
    stage is expanded by default, keeping the card compact in chat.
    """
    status_label = _COUNCIL_STATUS_LABEL_ZH.get(run.status, run.status.value)
    header_template = (
        "green" if run.status == CouncilStatus.COMPLETED
        else "red" if run.status == CouncilStatus.FAILED
        else "indigo"
    )

    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": f"**议题:** {redact_sensitive(run.question)[:300]}",
        }
    ]

    if run.error:
        elements.append({"tag": "markdown", "content": f"**错误:** {redact_sensitive(run.error)[:500]}"})

    elements.append({"tag": "hr"})

    # Determine which stage to expand (latest non-empty stage)
    stage_data = [
        ("1", "独立意见", _format_council_responses(run)),
        ("2", "匿名互评", _format_council_reviews(run)),
        ("3", "主席综合", _format_council_final(run)),
    ]
    last_active_idx = 0
    for idx, (_, _, content) in enumerate(stage_data):
        if content and not content.startswith("*等待"):
            last_active_idx = idx

    for idx, (stage_num, title, content) in enumerate(stage_data):
        elements.append({
            "tag": "collapsible_panel",
            "expanded": idx == last_active_idx,
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"阶段 {stage_num} — {title}",
                },
            },
            "border": {"color": "grey"},
            "body": {
                "elements": [
                    {"tag": "markdown", "content": content or "*等待中*"},
                ],
            },
        })

    elements.append({
        "tag": "markdown",
        "content": f"`council: {run.run_id[:12]}...` · {status_label}",
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title=f"🧭 Slock Council — {status_label}",
        header_template=header_template,
        elements=elements,
        mobile_optimize=False,
    )


# ------------------------------------------------------------------
# Internal constants
# ------------------------------------------------------------------

_STATUS_ICON_MAP: dict[AgentStatus, str] = {
    AgentStatus.IDLE: "🟢",
    AgentStatus.WAKING: "🟡",
    AgentStatus.THINKING: "🟡",
    AgentStatus.RUNNING: "🔵",
    AgentStatus.CHECKING: "🔵",
    AgentStatus.SENDING: "⚪",
    AgentStatus.MOVING: "🔶",
    AgentStatus.DISCUSSING: "💬",
    AgentStatus.PENDING_DISCUSSION: "⏳",
}

# Background color mapping: use the Single Source of Truth from models.py
# (Legacy _STATUS_BG_COLOR_MAP removed — use AGENT_STATUS_BG_COLOR_MAP directly)

_TASK_STATUS_ICONS: dict[TaskStatus, str] = {
    TaskStatus.TODO: "⬜",
    TaskStatus.IN_PROGRESS: "🔵",
    TaskStatus.IN_REVIEW: "🟡",
    TaskStatus.DONE: "✅",
}

# Background color mapping for task board Kanban columns (Feishu card background_style)
_TASK_STATUS_BG_COLOR_MAP: dict[TaskStatus, str] = {
    TaskStatus.TODO: "grey",
    TaskStatus.IN_PROGRESS: "blue",
    TaskStatus.IN_REVIEW: "yellow",
    TaskStatus.DONE: "green",
}

_COUNCIL_STATUS_LABEL_ZH: dict[CouncilStatus, str] = {
    CouncilStatus.STARTING: "准备中",
    CouncilStatus.STAGE1_RUNNING: "独立作答中",
    CouncilStatus.STAGE1_DONE: "独立意见完成",
    CouncilStatus.STAGE2_RUNNING: "匿名互评中",
    CouncilStatus.STAGE2_DONE: "匿名互评完成",
    CouncilStatus.STAGE3_RUNNING: "主席综合中",
    CouncilStatus.COMPLETED: "已完成",
    CouncilStatus.FAILED: "失败",
}


def _build_slock_group_jump_button(channel_id: str) -> dict:
    return apply_compact_style(
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "进入 Slock 群"},
            "type": "primary",
            "multi_url": _build_chat_multi_url(channel_id),
        }
    )


def _build_chat_multi_url(chat_id: str) -> dict:
    safe_chat_id = quote(str(chat_id or "").strip(), safe="")
    https = f"https://applink.feishu.cn/client/chat/open?openChatId={safe_chat_id}"
    native = f"lark://applink/client/chat/open?openChatId={safe_chat_id}"
    return {
        "url": https,
        "pc_url": https,
        "android_url": native,
        "ios_url": native,
    }


def _build_callback_button(
    text: str,
    action: str,
    *,
    channel_id: str = "",
    project_id: str = "",
    button_type: str = "default",
    extra_value: dict | None = None,
) -> dict:
    value = {"action": action, "channel_id": channel_id}
    if project_id:
        value["project_id"] = project_id
    if extra_value:
        value.update(extra_value)
    return apply_compact_style(
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
            "type": button_type,
            "value": value,
            "behaviors": [{"type": "callback", "value": value}],
        }
    )


def build_console_card(
    *,
    channel_id: str = "",
    agents: list[str] | None = None,
    tasks: list[dict] | None = None,
    team_name: str = "",
) -> dict:
    """Build the interactive console card for /slock root command.

    Uses collapsible panels for mobile-friendly layout:
    - Task Management (high-frequency): expanded by default
    - Agent/Team/Advanced (lower-frequency): collapsed by default
    """
    agents = agents or []
    tasks = tasks or []

    elements: list[dict] = []

    # Header info
    if team_name:
        elements.append({
            "tag": "markdown",
            "content": f"**团队:** {team_name}  |  **Agents:** {len(agents)}  |  **任务:** {len(tasks)}",
        })
        elements.append({"tag": "hr"})

    # Group 1: Task Management (high-frequency, expanded by default)
    task_buttons = [
        _build_callback_button("📌 分配任务", "slock_cmd_task_assign", channel_id=channel_id),
        _build_callback_button("📋 任务列表", "slock_cmd_task_list", channel_id=channel_id),
        _build_callback_button("📊 任务状态", "slock_cmd_task_status", channel_id=channel_id),
    ]
    elements.append({
        "tag": "collapsible_panel",
        "expanded": True,
        "header": {"title": {"tag": "markdown", "content": "📝 **任务管理**"}},
        "vertical_spacing": "8px",
        "elements": build_responsive_layout(task_buttons),
    })

    # Group 2: Agent Management (collapsed)
    agent_buttons = [
        _build_callback_button("➕ 新建角色", "slock_cmd_new_role", channel_id=channel_id),
        _build_callback_button("📋 角色列表", "slock_cmd_role_list", channel_id=channel_id),
        _build_callback_button("ℹ️ 角色详情", "slock_cmd_role_info", channel_id=channel_id),
        _build_callback_button("🗑️ 移除角色", "slock_cmd_role_remove", channel_id=channel_id),
    ]
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "🤖 **Agent 管理**"}},
        "vertical_spacing": "8px",
        "elements": build_responsive_layout(agent_buttons),
    })

    # Group 3: Team Management (collapsed)
    team_buttons = [
        _build_callback_button("📋 团队列表", "slock_cmd_team_list", channel_id=channel_id),
        _build_callback_button("📊 团队状态", "slock_cmd_team_status", channel_id=channel_id),
        _build_callback_button("⚠️ 解散团队", "slock_cmd_dissolve_team", channel_id=channel_id, button_type="danger"),
    ]
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "👥 **团队管理**"}},
        "vertical_spacing": "8px",
        "elements": build_responsive_layout(team_buttons),
    })

    # Group 4: Advanced Features (collapsed)
    advanced_buttons = [
        _build_callback_button("💬 发起讨论", "slock_cmd_discuss", channel_id=channel_id),
        _build_callback_button("🏛️ Council", "slock_cmd_council", channel_id=channel_id),
        _build_callback_button("🧠 记忆管理", "slock_cmd_memory", channel_id=channel_id),
        _build_callback_button("📊 状态面板", "slock_cmd_status", channel_id=channel_id),
    ]
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "⚡ **高级功能**"}},
        "vertical_spacing": "8px",
        "elements": build_responsive_layout(advanced_buttons),
    })

    return _build_card_wrapper(
        header_title="🎛️ Slock 控制台",
        header_template="indigo",
        header_subtitle="点击按钮快速执行命令",
        elements=elements,
        mobile_optimize=False,
    )


def build_error_suggestion_card(
    error_msg: str,
    suggestions: list[dict],
    *,
    channel_id: str = "",
) -> dict:
    """Build an error card with clickable correction suggestion buttons.

    Args:
        error_msg: The error description to display.
        suggestions: List of dicts with 'label' (button text) and 'command' (full command to execute).
        channel_id: Channel context for button callbacks.
    """
    elements: list[dict] = []

    # Error message
    elements.append({
        "tag": "markdown",
        "content": f"⚠️ **错误:** {error_msg}",
    })

    if suggestions:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": "💡 **建议修正:**",
        })

        suggestion_buttons = []
        for s in suggestions:
            label = s.get("label", "")
            command = s.get("command", "")
            suggestion_buttons.append(
                _build_callback_button(
                    f"▶ {label}",
                    "slock_execute_command",
                    channel_id=channel_id,
                    extra_value={"command": command},
                )
            )
        elements.extend(build_responsive_layout(suggestion_buttons))

    return _build_card_wrapper(
        header_title="❌ 命令错误",
        header_template="red",
        elements=elements,
        mobile_optimize=False,
    )


def build_dissolve_confirm_card(
    team_name: str,
    *,
    channel_id: str = "",
) -> dict:
    """Build a confirmation card before team dissolution."""
    elements: list[dict] = []

    elements.append({
        "tag": "markdown",
        "content": f"⚠️ 确认要解散团队 **{team_name}** 吗？\n\n此操作将移除所有 Agent 绑定和任务分配。",
    })

    elements.append({"tag": "hr"})

    buttons = [
        _build_callback_button(
            "✅ 确认解散",
            "slock_dissolve_confirm",
            channel_id=channel_id,
            button_type="danger",
            extra_value={"team_name": team_name},
        ),
        _build_callback_button(
            "❌ 取消",
            "slock_dissolve_cancel",
            channel_id=channel_id,
        ),
    ]
    elements.extend(build_responsive_layout(buttons))

    return _build_card_wrapper(
        header_title="⚠️ 解散团队确认",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )


def build_dissolve_undo_card(
    snapshot_id: str,
    *,
    channel_id: str = "",
    ttl: int = 30,
) -> dict:
    """Build a success card with undo button after team dissolution."""
    elements: list[dict] = []

    elements.append({
        "tag": "markdown",
        "content": f"✅ 团队已解散。\n\n⏱️ 你有 **{ttl}秒** 可以撤销此操作。",
    })

    elements.append({"tag": "hr"})

    buttons = [
        _build_callback_button(
            "↩️ 撤销解散",
            "slock_dissolve_undo",
            channel_id=channel_id,
            button_type="primary",
            extra_value={"snapshot_id": snapshot_id},
        ),
    ]
    elements.extend(build_responsive_layout(buttons))

    return _build_card_wrapper(
        header_title="✅ 团队已解散",
        header_template="green",
        elements=elements,
        mobile_optimize=False,
    )


# ---------------------------------------------------------------------------
# Memory & Role switch cards
# ---------------------------------------------------------------------------


def build_memory_display_card(memory: SlockMemory, agent_name: str = "Agent") -> dict:
    """Build a card displaying an agent's L1 memory (role/key_knowledge/active_context)."""
    elements: list[dict] = []

    # Role section (expanded by default)
    role_text = memory.role.strip() if memory.role else "(未定义)"
    role_text = redact_sensitive(role_text)
    elements.append({
        "tag": "collapsible_panel",
        "expanded": True,
        "header": {"title": {"tag": "markdown", "content": "**📋 角色定义**"}},
        "vertical_spacing": "8px",
        "elements": [{"tag": "markdown", "content": role_text[:800]}],
    })

    # Key Knowledge section (collapsed)
    kk_text = memory.key_knowledge.strip() if memory.key_knowledge else "(空)"
    kk_text = redact_sensitive(kk_text)
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "**🔑 关键知识**"}},
        "vertical_spacing": "8px",
        "elements": [{"tag": "markdown", "content": kk_text[:1500]}],
    })

    # Active Context section (collapsed)
    ctx_text = memory.active_context.strip() if memory.active_context else "(空)"
    ctx_text = redact_sensitive(ctx_text)
    if len(ctx_text) > 1500:
        ctx_text = ctx_text[:1500] + "\n..."
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "**💭 活跃上下文**"}},
        "vertical_spacing": "8px",
        "elements": [{"tag": "markdown", "content": ctx_text}],
    })

    return _build_card_wrapper(
        header_title=f"🧠 Agent 记忆 — {agent_name}",
        header_template="turquoise",
        elements=elements,
        mobile_optimize=False,
    )


def build_memory_manage_card(
    memory: "SlockMemory",
    agent_name: str,
    agent_id: str,
    can_edit: bool = False,
    channel_id: str = "",
) -> dict:
    """Build memory management card with action buttons.

    Extends build_memory_display_card with edit/clear/archive buttons.
    Only shows edit buttons if can_edit is True (owner/admin).
    """
    elements: list[dict] = []

    # Role section (expanded by default)
    role_text = memory.role.strip() if memory.role else "(未定义)"
    role_text = redact_sensitive(role_text)
    elements.append({
        "tag": "collapsible_panel",
        "expanded": True,
        "header": {"title": {"tag": "markdown", "content": "**📋 角色定义**"}},
        "vertical_spacing": "8px",
        "elements": [{"tag": "markdown", "content": role_text[:800]}],
    })

    # Key Knowledge section (collapsed)
    kk_text = memory.key_knowledge.strip() if memory.key_knowledge else "(空)"
    kk_text = redact_sensitive(kk_text)
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "**🔑 关键知识**"}},
        "vertical_spacing": "8px",
        "elements": [{"tag": "markdown", "content": kk_text[:1500]}],
    })

    # Active Context section (collapsed)
    ctx_text = memory.active_context.strip() if memory.active_context else "(空)"
    ctx_text = redact_sensitive(ctx_text)
    if len(ctx_text) > 1500:
        ctx_text = ctx_text[:1500] + "\n..."
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "**💭 活跃上下文**"}},
        "vertical_spacing": "8px",
        "elements": [{"tag": "markdown", "content": ctx_text}],
    })

    # Archived Context section (collapsed)
    arch_text = memory.archived_context.strip() if memory.archived_context else "(空)"
    arch_text = redact_sensitive(arch_text)
    if len(arch_text) > 1500:
        arch_text = arch_text[:1500] + "\n..."
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "markdown", "content": "**📦 归档上下文**"}},
        "vertical_spacing": "8px",
        "elements": [{"tag": "markdown", "content": arch_text}],
    })

    # Action buttons (only for authorized users)
    if can_edit:
        elements.append({"tag": "hr"})
        action_buttons: list[dict] = []
        action_buttons.append(
            _build_callback_button(
                "✏️ 编辑角色定义",
                "slock_memory_edit_role",
                channel_id=channel_id,
                button_type="default",
                extra_value={"agent_id": agent_id},
            )
        )
        action_buttons.append(
            _build_callback_button(
                "🗑️ 清空活跃上下文",
                "slock_memory_clear_context",
                channel_id=channel_id,
                button_type="danger",
                extra_value={"agent_id": agent_id},
            )
        )
        elements.extend(build_responsive_layout(action_buttons))

    return _build_card_wrapper(
        header_title=f"🧠 记忆管理 — {agent_name}",
        header_template="turquoise",
        elements=elements,
        mobile_optimize=False,
    )


def build_conclusion_notification_card(
    conclusion_preview: str,
    participants: list[str],
    affected_agents: list[str] | None = None,
    skipped_agents: list[str] | None = None,
    detection_timed_out: bool = False,
    channel_id: str = "",
) -> dict:
    """Build a lightweight notification card after discussion conclusion is persisted.

    Shows the first 100 chars of the conclusion and the participant list.
    """
    preview = conclusion_preview[:100]
    if len(conclusion_preview) > 100:
        preview += "..."

    participants_text = "、".join(participants) if participants else "(无)"
    affected_text = "、".join(affected_agents or participants) if (affected_agents or participants) else "(无)"
    skipped = [agent for agent in (skipped_agents or []) if agent]
    has_conflict = bool(skipped)
    header_title = "💬 讨论结论已持久化"
    header_template = "green"
    if has_conflict:
        header_title = "💬 讨论结论已持久化（存在冲突）"
        header_template = "orange"

    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": (
                f"**📝 结论摘要:**\n{preview}\n\n"
                f"**👥 参与者:** {participants_text}\n\n"
                f"**✅ 已同步:** {affected_text}"
            ),
        },
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": "✅ 讨论结论已同步至参与 Agent 的 L1 记忆" if not has_conflict else "⚠️ 部分同步完成，部分 Agent 存在知识冲突",
        },
    ]

    if has_conflict:
        elements.append({
            "tag": "markdown",
            "content": f"⚠️ **知识冲突:** {'、'.join(skipped)}",
        })
        elements.extend(build_responsive_layout([
            _build_callback_button(
                "查看冲突",
                "slock_conclusion_conflict_view",
                channel_id=channel_id,
                button_type="primary",
                extra_value={"action": "view_conflict", "agents": skipped},
            ),
            _build_callback_button(
                "强制覆盖",
                "slock_conclusion_force_override",
                channel_id=channel_id,
                button_type="danger",
                extra_value={"action": "force_override", "agents": skipped},
            ),
        ]))

    if detection_timed_out:
        elements.append({
            "tag": "markdown",
            "content": "⏱️ 语义冲突检测超时，已跳过 LLM 校验。",
            "text_size": "notation",
        })

    return _build_card_wrapper(
        header_title=header_title,
        header_template=header_template,
        elements=elements,
        mobile_optimize=False,
    )


def build_role_switch_card(
    roles: list[str],
    agent_id: str,
    channel_id: str = "",
    project_id: str = "",
) -> dict:
    """Build a card with role selection buttons for switching an agent's role."""
    elements: list[dict] = []

    elements.append({"tag": "markdown", "content": "请选择要切换到的角色："})

    role_buttons = [
        _build_callback_button(
            f"🎭 {role}",
            "slock_confirm_switch_role",
            channel_id=channel_id,
            project_id=project_id,
            button_type="default",
            extra_value={"agent_id": agent_id, "target_role": role},
        )
        for role in roles
    ]
    elements.extend(build_responsive_layout(role_buttons))

    return _build_card_wrapper(
        header_title="🎭 切换角色",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )


def _wrap_text(text: str, width: int = 80) -> str:
    """Wrap lines longer than *width* characters by inserting newlines."""
    lines = text.split("\n")
    wrapped: list[str] = []
    for line in lines:
        while len(line) > width:
            wrapped.append(line[:width])
            line = line[width:]
        wrapped.append(line)
    return "\n".join(wrapped)


def build_escalation_card(
    escalation: EscalationRequest,
    *,
    channel_id: str = "",
    timeout_minutes: Optional[int] = None,
) -> dict:
    """Build an escalation card requesting admin intervention.

    Shows the agent name, severity level, reason, and resolution option buttons.
    """
    level_icons = {
        EscalationLevel.WARNING: "⚠️",
        EscalationLevel.BLOCKED: "🚫",
        EscalationLevel.CRITICAL: "🔴",
    }
    level_colors = {
        EscalationLevel.WARNING: "yellow",
        EscalationLevel.BLOCKED: "orange",
        EscalationLevel.CRITICAL: "red",
    }

    icon = level_icons.get(escalation.level, "⚠️")
    header_color = level_colors.get(escalation.level, "orange")
    header_title = f"{icon} 升级告警: {escalation.agent_name or 'Agent'}"

    elements: list[dict] = []

    # Severity and reason (redact sensitive info before rendering)
    safe_reason = redact_sensitive(escalation.reason)
    elements.append({
        "tag": "markdown",
        "content": (
            f"**级别:** {escalation.level.value.upper()}\n"
            f"**代理:** {escalation.agent_name} (`{escalation.agent_id}`)\n"
            f"**原因:** {safe_reason}"
        ),
    })

    # Context details (truncated + redacted)
    if escalation.context:
        context_display = redact_sensitive(escalation.context[:500])
        if len(escalation.context) > 500:
            context_display += "\n..."
        context_display = _wrap_text(context_display)
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": f"**上下文:**\n{context_display}",
        })

    # Task reference
    if escalation.task_id:
        elements.append({
            "tag": "markdown",
            "content": f"**任务:** `{escalation.task_id}`",
        })

    # Timeout hint
    if timeout_minutes is not None:
        elements.append({
            "tag": "markdown",
            "content": f"⏰ 此升级将在 {timeout_minutes} 分钟后自动中止",
            "text_size": "notation",
        })

    elements.append({"tag": "hr"})

    # Resolution option buttons
    option_buttons: list[dict] = []
    default_options = escalation.options or ["重试", "跳过", "中止"]
    for option in default_options:
        value = {
            "action": "slock_escalation_resolve",
            "escalation_id": escalation.escalation_id,
            "resolution": option,
            "channel_id": channel_id,
        }
        btn_type = "danger" if option in ABORT_OPTIONS else "default"
        option_buttons.append(apply_compact_style({
            "tag": "button",
            "text": {"tag": "plain_text", "content": option},
            "type": btn_type,
            "value": value,
            "behaviors": [{"type": "callback", "value": value}],
        }))

    if option_buttons:
        elements.extend(build_responsive_layout(option_buttons))

    return _build_card_wrapper(
        header_title=header_title,
        header_template=header_color,
        elements=elements,
        mobile_optimize=False,
    )


def build_resolved_escalation_card(
    escalation: EscalationRequest,
    *,
    resolved_by: str = "",
    resolution: str = "",
    resolved_at: Optional[float] = None,
    channel_id: str = "",
) -> dict:
    """Build an escalation card in resolved state — buttons removed, status shown.

    Args:
        escalation: The original EscalationRequest.
        resolved_by: Display name or ID of the operator who resolved it.
        resolution: The chosen resolution (e.g. Retry/Skip/Abort).
        resolved_at: Timestamp of resolution (epoch seconds).
        channel_id: Optional channel identifier.
    """
    import time as _time

    level_icons = {
        EscalationLevel.WARNING: "⚠️",
        EscalationLevel.BLOCKED: "🚫",
        EscalationLevel.CRITICAL: "🔴",
    }

    icon = level_icons.get(escalation.level, "⚠️")
    header_title = f"{icon} 升级告警: {escalation.agent_name or 'Agent'} [已解决]"

    elements: list[dict] = []

    # Original severity and reason
    elements.append({
        "tag": "markdown",
        "content": (
            f"**级别:** {escalation.level.value.upper()}\n"
            f"**代理:** {escalation.agent_name} (`{escalation.agent_id}`)\n"
            f"**原因:** {redact_sensitive(escalation.reason)}"
        ),
    })

    # Context details (truncated) — keep for reference
    if escalation.context:
        context_display = redact_sensitive(escalation.context[:500])
        if len(escalation.context) > 500:
            context_display += "\n..."
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": f"**上下文:**\n```\n{context_display}\n```",
        })

    # Task reference
    if escalation.task_id:
        elements.append({
            "tag": "markdown",
            "content": f"**任务:** `{escalation.task_id}`",
        })

    elements.append({"tag": "hr"})

    # Resolution status (replaces buttons)
    ts = resolved_at or _time.time()
    from datetime import datetime
    time_str = datetime.fromtimestamp(ts, tz=_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
    operator_display = resolved_by or "未知"

    elements.append({
        "tag": "markdown",
        "content": f"✅ **已解决:** {resolution}，由 {operator_display} 处理\n📅 {time_str}",
    })

    return _build_card_wrapper(
        header_title=header_title,
        header_template="green",
        elements=elements,
        mobile_optimize=False,
    )


def build_crash_recovery_card(
    recovered_tasks: list[SlockTask],
    *,
    channel_id: str = "",
) -> dict:
    """Build a notification card for tasks recovered after crash/restart.

    Shows the list of tasks that were downgraded from IN_PROGRESS/IN_REVIEW
    back to TODO during channel activation.
    """
    elements: list[dict] = []

    elements.append({
        "tag": "markdown",
        "content": "系统重启后发现以下任务处于中间状态，已自动降级为 **待办**：",
    })

    # Task list
    task_lines = []
    for task in recovered_tasks[:20]:  # Cap display at 20
        task_id_short = task.task_id[:8]
        content_preview = task.content[:60] + ("..." if len(task.content) > 60 else "")
        task_lines.append(f"• `{task_id_short}` {content_preview}")

    if len(recovered_tasks) > 20:
        task_lines.append(f"• ... 还有 {len(recovered_tasks) - 20} 个任务")

    elements.append({
        "tag": "markdown",
        "content": "\n".join(task_lines),
    })

    elements.append({"tag": "hr"})
    elements.append({
        "tag": "markdown",
        "content": "💡 这些任务已重置为待办状态，可通过 `/task` 命令重新分配或触发执行。",
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title="⚡ 系统恢复通知",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )


def build_review_degradation_card(
    degraded_tasks: list[SlockTask],
    *,
    channel_id: str = "",
) -> dict:
    """Build a notification card for IN_REVIEW tasks degraded to TODO on restart.

    These tasks lost their review context and need re-review.
    """
    elements: list[dict] = []

    elements.append({
        "tag": "markdown",
        "content": "以下任务在重启前处于 **审阅中** 状态，审阅上下文已丢失，已降级为 **待办**：",
    })

    task_lines = []
    for task in degraded_tasks[:20]:
        task_id_short = task.task_id[:8]
        content_preview = task.content[:60] + ("..." if len(task.content) > 60 else "")
        task_lines.append(f"• `{task_id_short}` {content_preview}")

    if len(degraded_tasks) > 20:
        task_lines.append(f"• ... 还有 {len(degraded_tasks) - 20} 个任务")

    elements.append({
        "tag": "markdown",
        "content": "\n".join(task_lines),
    })

    elements.append({"tag": "hr"})
    elements.append({
        "tag": "markdown",
        "content": "⚠️ 这些任务需要重新进入审阅流程。可使用 `/task review <ID>` 重新提交审阅。",
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title="⚠️ 审阅状态降级通知",
        header_template="yellow",
        elements=elements,
        mobile_optimize=False,
    )


# ---------------------------------------------------------------------------
# Task 14-18: UX Card Templates
# ---------------------------------------------------------------------------


def build_command_hub_card(*, channel_id: str = "") -> dict:
    """Build the /slock entry-point hub card with 4 grouped action panels.

    Groups:
    1. Agent 管理 (create role, list roles, role info, remove role)
    2. 任务管理 (assign task, list tasks, task status)
    3. 团队管理 (create team, list teams, dissolve team)
    4. 系统控制 (status, stop, council, help)
    """

    def _hub_btn(label: str, command: str, *, style: str = "default") -> dict:
        value = {"action": "slock_hub_cmd", "cmd": command, "channel_id": channel_id}
        return apply_compact_style({
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": style,
            "value": value,
            "behaviors": [{"type": "callback", "value": value}],
        }, skip_compact=True)

    groups = [
        {
            "title": "🤖 Agent 管理",
            "buttons": [
                _hub_btn("➕ 新建角色", "/new-role"),
                _hub_btn("📋 角色列表", "/role list"),
                _hub_btn("ℹ️ 角色详情", "/role info"),
                _hub_btn("🗑 移除角色", "/role remove", style="danger"),
            ],
        },
        {
            "title": "📝 任务管理",
            "buttons": [
                _hub_btn("📌 分配任务", "/task assign"),
                _hub_btn("📋 任务列表", "/task list"),
                _hub_btn("📊 任务状态", "/task status"),
            ],
        },
        {
            "title": "👥 团队管理",
            "buttons": [
                _hub_btn("➕ 新建团队", "/new-team"),
                _hub_btn("📋 团队列表", "/team list"),
                _hub_btn("🗑 解散团队", "/team dissolve", style="danger"),
            ],
        },
        {
            "title": "⚙️ 系统控制",
            "buttons": [
                _hub_btn("📊 状态面板", "/slock status"),
                _hub_btn("🏛 Council", "/council"),
                _hub_btn("⏹ 停止引擎", "/slock stop", style="danger"),
                _hub_btn("❓ 帮助", "/slock help"),
            ],
        },
    ]

    elements: list[dict] = []
    for group in groups:
        # Group title
        elements.append({
            "tag": "markdown",
            "content": f"**{group['title']}**",
        })
        # Buttons via responsive layout (mobile-friendly vertical stacking for >2 buttons)
        elements.extend(build_responsive_layout(group["buttons"], mobile_force_vertical=True))
        # Divider between groups (except last)
        if group != groups[-1]:
            elements.append({"tag": "hr"})

    return _build_card_wrapper(
        header_title="🎛 Slock 命令面板",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )


def build_command_panel_card(*, channel_id: str = "", project_id: str = "") -> dict:
    """Build a compact command panel with core actions and a 'more' button.

    Primary level shows 4 quick-action buttons + expand trigger.
    Extended forms are served via build_command_panel_extended_card().
    """
    elements: list[dict] = []

    # Primary quick-action buttons (first screen, max 3)
    primary_buttons = [
        _build_callback_button("🏠 查看团队", "slock_cmd_team_list", channel_id=channel_id, project_id=project_id, button_type="default"),
        _build_callback_button("🎭 查看角色", "slock_cmd_role_list", channel_id=channel_id, project_id=project_id, button_type="default"),
        _build_callback_button("📋 任务面板", "slock_cmd_task_list", channel_id=channel_id, project_id=project_id, button_type="default"),
    ]
    elements.extend(build_responsive_layout(primary_buttons))

    # Secondary buttons in collapsible panel (default collapsed for mobile)
    secondary_buttons = [
        _build_callback_button("🧠 查看记忆", "slock_cmd_memory", channel_id=channel_id, project_id=project_id, button_type="default"),
        _build_callback_button("🗣 发起讨论", "slock_cmd_discuss", channel_id=channel_id, project_id=project_id, button_type="primary"),
    ]
    collapsible_elements = build_responsive_layout(secondary_buttons)
    # Add role naming guidance inside collapsible panel
    collapsible_elements.append({
        "tag": "markdown",
        "content": "📌 **角色名语法提示**: 带空格的角色名请使用 `@role` 或双引号包裹（如 `\"Senior Coder\"`），避免解析歧义。",
    })
    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "plain_text", "content": "📂 更多快捷操作"}},
        "vertical_spacing": "8px",
        "elements": collapsible_elements,
    })

    # Expand button for extended operations
    elements.append({"tag": "hr"})
    elements.extend(build_responsive_layout([
        _build_callback_button(
            "⚙️ 更多操作...",
            "slock_cmd_panel_extended",
            channel_id=channel_id,
            project_id=project_id,
            button_type="default",
        ),
    ]))

    # Bottom hint
    elements.append({
        "tag": "markdown",
        "content": "<font color='grey'>💡 也可直接输入命令：/team、/role、/task、/council、/discuss、/memory</font>",
    })

    return _build_card_wrapper(
        header_title="\U0001f4cb Slock 命令面板",
        header_template="blue",
        elements=elements,
        mobile_optimize=False,
    )


def build_command_panel_extended_card(*, channel_id: str = "", project_id: str = "") -> dict:
    """Build the extended command panel with action-based input forms.

    This is the second-level card triggered by '更多操作' button.
    Contains team creation, role creation, and council forms.
    Uses standard 'action' elements (not 'form') for Feishu Schema 2.0 compatibility.
    """
    elements: list[dict] = []

    # Team creation action
    elements.append({"tag": "markdown", "content": "**🏠 创建团队**"})
    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "input",
                "name": "team_name",
                "placeholder": {"tag": "plain_text", "content": "输入团队名称"},
                "width": "fill",
            },
            _build_callback_button(
                "创建团队",
                "slock_form_new_team",
                channel_id=channel_id,
                project_id=project_id,
                button_type="primary",
            ),
        ],
    })

    # Role creation action
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": "**🎭 创建角色**"})
    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "input",
                "name": "role_name",
                "placeholder": {"tag": "plain_text", "content": "输入角色名称（如: coder-小明）"},
                "width": "fill",
            },
            _build_callback_button(
                "创建角色",
                "slock_form_new_role",
                channel_id=channel_id,
                project_id=project_id,
                button_type="primary",
            ),
        ],
    })

    # Council action
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": "**🧑\u200d⚖️ Council 评审**"})
    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "input",
                "name": "council_topic",
                "placeholder": {"tag": "plain_text", "content": "输入评审议题"},
                "width": "fill",
            },
            _build_callback_button(
                "发起评审",
                "slock_form_council",
                channel_id=channel_id,
                project_id=project_id,
                button_type="primary",
            ),
        ],
    })

    # Bottom hint
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "markdown",
        "content": "<font color='grey'>💡 返回主面板：输入 /slock</font>",
    })

    return _build_card_wrapper(
        header_title="⚙️ Slock 扩展操作",
        header_template="blue",
        elements=elements,
        mobile_optimize=False,
    )


def build_error_suggestion_card(  # noqa: F811
    user_input: str,
    suggestions: list[str],
    *,
    channel_id: str = "",
    project_id: str = "",
) -> dict:
    """Build an error card with clickable correction suggestion buttons."""
    elements: list[dict] = []

    # Truncate and redact user input for display
    display_input = redact_sensitive(user_input)
    if len(display_input) > 50:
        display_input = display_input[:50] + "..."

    elements.append({
        "tag": "markdown",
        "content": f"无法识别您的输入：`{display_input}`",
    })

    # Render suggestions as clickable buttons
    if suggestions:
        elements.append({"tag": "markdown", "content": "**您是否想要：**"})
        suggestion_buttons = [
            _build_callback_button(
                f"➡️ {suggestion}",
                "slock_cmd_fix",
                channel_id=channel_id,
                project_id=project_id,
                button_type="default",
                extra_value={"fix_command": suggestion},
            )
            for suggestion in suggestions[:5]
        ]
        elements.extend(build_responsive_layout(suggestion_buttons))

    elements.append({"tag": "hr"})
    elements.append({
        "tag": "markdown",
        "content": "<font color='grey'>💡 也可以输入 /help 查看所有可用命令</font>",
    })

    return _build_card_wrapper(
        header_title="💡 无法识别指令",
        header_template="wathet",
        elements=elements,
        mobile_optimize=False,
    )


def build_cmd_arg_error_card(
    user_input: str,
    usage_hint: str,
    suggestions: list[str],
    *,
    channel_id: str = "",
    project_id: str = "",
    prefix_label: str = "",
) -> dict:
    """Build an error card for missing command arguments with fix suggestion buttons.

    Args:
        user_input: The original command text the user typed (e.g. "/role remove").
        usage_hint: A usage example line (e.g. "用法: `/role remove <名称>`").
        suggestions: List of corrected command strings (e.g. ["/role remove Alice"]).
        channel_id: Feishu chat/channel ID for callback routing.
        project_id: Project ID for callback routing.
        prefix_label: Optional markdown annotation displayed above button group.

    Returns:
        Feishu card dict (schema 2.0) with clickable suggestion buttons.
    """
    elements: list[dict] = []

    display_input = redact_sensitive(user_input)
    if len(display_input) > 80:
        display_input = display_input[:80] + "..."

    elements.append({
        "tag": "markdown",
        "content": f"⚠️ 参数缺失：`{display_input}`",
    })

    # Visual separator before usage hint
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "markdown",
        "content": usage_hint,
    })

    if suggestions:
        if prefix_label:
            elements.append({"tag": "markdown", "content": prefix_label})
        else:
            elements.append({"tag": "markdown", "content": "**可选角色：**"})
        suggestion_buttons = [
            _build_callback_button(
                _truncate_button_label(suggestion),
                "slock_cmd_fix",
                channel_id=channel_id,
                project_id=project_id,
                button_type="default",
                extra_value={"fix_command": suggestion},
            )
            for suggestion in suggestions[:5]
        ]
        elements.extend(build_responsive_layout(suggestion_buttons))

    elements.append({"tag": "hr"})
    has_placeholder = any("<" in s and ">" in s for s in suggestions)
    note_text = (
        "💡 点击按钮可复制命令模板，请替换 `<...>` 占位符后发送"
        if has_placeholder
        else "💡 点击按钮可直接执行修正后的命令"
    )
    elements.append({
        "tag": "markdown",
        "content": f"<font color='grey'>{note_text}</font>",
    })

    return _build_card_wrapper(
        header_title="⚠️ 命令参数缺失",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )


def _truncate_button_label(suggestion: str) -> str:
    """Extract core parameter from a command suggestion for button display (≤20 chars).

    Given "/role remove Alice", returns "➡️ Alice".
    Given "/task assign fix-bug Coder", returns "➡️ Coder".
    """
    parts = suggestion.strip().split()
    # Skip command prefix parts (start with /)
    core_parts = [p for p in parts if not p.startswith("/")]
    # Use the last meaningful token as the button label (typically the role/agent name)
    label = core_parts[-1] if core_parts else suggestion
    label = f"➡️ {label}"
    if len(label) > 20:
        label = label[:19] + "…"
    return label


def _truncate_dynamic_label(text: str, max_len: int = 20) -> str:
    """Truncate dynamic button label to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


# Backward-compatible alias
build_role_arg_error_card = build_cmd_arg_error_card


def build_confirm_cancel_card(
    title: str,
    description: str,
    *,
    severity: str = "warning",
    confirm_action: str = "slock_confirm",
    cancel_action: str = "slock_cancel",
    channel_id: str = "",
    extra_value: dict | None = None,
) -> dict:
    """Build a reusable confirmation dialog card."""
    _severity_map: dict[str, dict[str, str]] = {
        "info": {"template": "blue", "button_type": "primary"},
        "warning": {"template": "orange", "button_type": "primary"},
        "danger": {"template": "red", "button_type": "danger"},
    }
    sev = _severity_map.get(severity, _severity_map["warning"])

    elements: list[dict] = []

    elements.append({"tag": "markdown", "content": description})

    # Confirm + Cancel buttons
    elements.extend(
        build_responsive_layout(
            [
                _build_callback_button(
                    "确认",
                    confirm_action,
                    channel_id=channel_id,
                    button_type=sev["button_type"],
                    extra_value=extra_value,
                ),
                _build_callback_button(
                    "取消",
                    cancel_action,
                    channel_id=channel_id,
                    button_type="default",
                    extra_value=extra_value,
                ),
            ]
        )
    )

    return _build_card_wrapper(
        header_title=title,
        header_template=sev["template"],
        elements=elements,
        mobile_optimize=False,
    )


def build_dissolve_confirm_card(  # noqa: F811
    team_name: str,
    *,
    channel_id: str = "",
) -> dict:
    """Build a team dissolution confirmation card with 30s undo window notice.

    Shows a danger-level confirmation with team name, warns about irreversibility,
    and indicates that an undo window will be available for 30 seconds after confirm.
    """
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": (
                f"⚠️ 确认解散团队 **{team_name}**？\n\n"
                "此操作将移除所有 Agent 绑定和角色分配。\n"
                "确认后有 **30 秒** 撤销窗口。"
            ),
        },
    ]

    elements.extend(
        build_responsive_layout([
            _build_callback_button(
                "🗑️ 确认解散",
                "slock_confirm_dissolve",
                channel_id=channel_id,
                button_type="danger",
                extra_value={"team_name": team_name},
            ),
            _build_callback_button(
                "取消",
                "slock_cancel_dissolve",
                channel_id=channel_id,
                button_type="default",
                extra_value={"team_name": team_name},
            ),
        ])
    )

    return _build_card_wrapper(
        header_title="🗑️ 解散团队确认",
        header_template="red",
        elements=elements,
        mobile_optimize=False,
    )


def build_council_detail_card(
    topic: str,
    opinions: list[dict],
    *,
    final_summary: str = "",
    channel_id: str = "",
    scores: list[dict] | None = None,
) -> dict:
    """Build an expanded council review detail card with collapsible opinions.

    Args:
        topic: The council review topic.
        opinions: List of dicts with keys: agent_name, emoji, role, opinion_text.
        final_summary: Optional synthesis summary.
        scores: Optional list of dicts with keys: agent_name, score (for ranking).
    """
    elements: list[dict] = []

    # Topic
    elements.append({"tag": "markdown", "content": f"**议题：** {topic}"})
    elements.append({"tag": "hr"})

    # Score ranking summary (if provided)
    if scores:
        sorted_scores = sorted(scores, key=lambda x: x.get("score", 0), reverse=True)
        ranking_lines = ["**📊 评分排名**"]
        for idx, s in enumerate(sorted_scores):
            medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"{idx + 1}."
            ranking_lines.append(f"{medal} {s.get('agent_name', 'Agent')} — {s.get('score', 0)}分")
        elements.append({"tag": "markdown", "content": "\n".join(ranking_lines)})
        elements.append({"tag": "hr"})

    # Each opinion in a collapsible_panel (with fallback to plain markdown)
    for idx, opinion in enumerate(opinions):
        agent_name = opinion.get("agent_name", "Agent")
        emoji = opinion.get("emoji", "\U0001f916")
        role = opinion.get("role", "")
        opinion_text = opinion.get("opinion_text", "")

        # Use collapsible_panel for expandable sections
        elements.append({
            "tag": "collapsible_panel",
            "expanded": idx == 0,  # First one expanded by default
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"{emoji} {agent_name} ({role})",
                },
            },
            "border": {"color": "grey"},
            "body": {
                "elements": [
                    {"tag": "markdown", "content": opinion_text[:2000] if opinion_text else "(无回答)"},
                ],
            },
        })

    # Final summary section
    if final_summary:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": f"**\U0001f4dd 综合评估**\n{final_summary}",
        })

    return _build_card_wrapper(
        header_title="\U0001f3db\ufe0f Council 评审详情",
        header_template="purple",
        elements=elements,
        mobile_optimize=False,
    )


def _build_agent_status_rows(agents_data: list[dict]) -> list[dict]:
    """Produce column_set elements with background_style color coding for agents.

    Args:
        agents_data: List of dicts with keys: name, emoji, status, role.

    Returns:
        A list of column_set card element dicts, one per agent.
    """
    # Derive string-based lookup from canonical AGENT_STATUS_BG_COLOR_MAP
    _str_bg_color_map: dict[str, str] = {s.value: c for s, c in AGENT_STATUS_BG_COLOR_MAP.items()}
    _str_bg_color_map["error"] = "red"  # Extra pseudo-state for error display
    _str_icon_map: dict[str, str] = {
        "idle": "\U0001f7e2",
        "waking": "\U0001f7e1",
        "thinking": "\U0001f7e1",
        "running": "\U0001f535",
        "checking": "\U0001f535",
        "sending": "\u26aa",
        "moving": "\U0001f536",
        "discussing": "\U0001f4ac",
        "pending_discussion": "\u23f3",
        "error": "\U0001f534",
    }

    rows: list[dict] = []
    for agent in agents_data:
        emoji = agent.get("emoji", "\U0001f916")
        name = agent.get("name", "Agent")
        status = agent.get("status", "idle").lower()
        role = agent.get("role", "")

        _str_bg_color_map.get(status, "grey")
        status_icon = _str_icon_map.get(status, "\u26aa")
        status_label = _STATUS_LABEL_ZH.get(status, status)

        # Single-line inline markdown — avoids column_set/flex_mode mobile overflow
        line = f"{emoji} **{name}** {status_icon} {status_label}"
        if role:
            line += f" · {role}"

        rows.append({
            "tag": "markdown",
            "content": line,
            "text_size": "normal",
        })
    return rows


def build_status_refresh_card(
    agents: list[dict],
    tasks_summary: dict,
    *,
    channel_id: str = "",
) -> dict:
    """Build a team status card with refresh button.

    Args:
        agents: List of dicts with keys: name, emoji, status, role.
        tasks_summary: Dict with keys: total, todo, in_progress, done.
    """
    elements: list[dict] = []

    # Agent status rows (colored column_set)
    if agents:
        elements.append({"tag": "markdown", "content": "**Agent 状态**"})
        elements.extend(_build_agent_status_rows(agents))

    elements.append({"tag": "hr"})

    # Task summary
    total = tasks_summary.get("total", 0)
    todo = tasks_summary.get("todo", 0)
    in_progress = tasks_summary.get("in_progress", 0)
    done = tasks_summary.get("done", 0)

    summary_content = (
        f"**任务概览**\n"
        f"  总计: **{total}** | 待办: **{todo}** | "
        f"进行中: **{in_progress}** | 已完成: **{done}**"
    )
    elements.append({"tag": "markdown", "content": summary_content})

    # Refresh button
    elements.extend(
        build_responsive_layout(
            [
                _build_callback_button(
                    "\U0001f504 刷新",
                    "slock_refresh_status",
                    channel_id=channel_id,
                ),
            ]
        )
    )

    return _build_card_wrapper(
        header_title="\U0001f4ca 团队状态",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )


def build_council_result_card(
    question: str,
    agents_answers: list[dict],
    rankings: list[dict],
    *,
    channel_id: str = "",
) -> dict:
    """Build a council result card with collapsible agent answers and rankings.

    Args:
        question: The original question posed to the council.
        agents_answers: List of dicts with keys: agent_name, answer, score.
        rankings: List of dicts with keys: rank, agent_name, score.
        channel_id: Optional channel identifier.
    """
    # Build ranking section
    ranking_lines = []
    for r in rankings:
        medal = {1: "\U0001f947", 2: "\U0001f948", 3: "\U0001f949"}.get(r.get("rank", 0), f"#{r.get('rank', '?')}")
        ranking_lines.append(f"{medal} **{r.get('agent_name', '?')}** \u2014 {r.get('score', 0):.1f}\u5206")

    ranking_text = "\n".join(ranking_lines) if ranking_lines else "\u65e0\u8bc4\u5206\u6570\u636e"

    # Build collapsible agent answer elements (sorted by ranking order with ordinal)
    # Create rank lookup from rankings
    _rank_by_name: dict[str, int] = {r.get("agent_name", ""): r.get("rank", 0) for r in rankings}
    agent_elements = []
    for idx, ans in enumerate(agents_answers, 1):
        agent_name = ans.get("agent_name", "?")
        score = ans.get("score", 0)
        rank = _rank_by_name.get(agent_name, idx)
        {1: "\U0001f947", 2: "\U0001f948", 3: "\U0001f949"}.get(rank, f"#{rank}")
        response_text = (ans.get("answer", "") or "\u65e0\u56de\u7b54")[:2000]
        agent_elements.append({
            "tag": "collapsible_panel",
            "expanded": False,
            "header": {"tag": "markdown", "content": f"\U0001f4dd **{agent_name}** (\u8bc4\u5206: {score:.1f}/10)"},
            "vertical_spacing": "8px",
            "elements": [
                {"tag": "markdown", "content": response_text},
            ],
        })

    card_elements: list[dict] = [
        {
            "tag": "markdown",
            "content": f"**\u95ee\u9898\uff1a** {question[:200]}",
        },
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": f"**\u8bc4\u5206\u6392\u540d**\n{ranking_text}",
        },
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": "**\u5404 Agent \u539f\u59cb\u56de\u7b54** (\u70b9\u51fb\u5c55\u5f00)",
        },
        *agent_elements,
    ]
    card = _build_card_wrapper(
        header_title="\U0001f4cb Council \u8bc4\u5ba1\u7ed3\u679c",
        header_template="blue",
        elements=card_elements,
        mobile_optimize=False,
    )
    return card


def build_chitchat_hint_card(
    original_message: str,
    *,
    channel_id: str = "",
    timestamp: Optional[float] = None,
) -> dict:
    """Build a hint card when a message is filtered as chitchat.

    Includes a 'force process' button that allows the user to override
    the chitchat classification and re-route the message to an agent.

    Args:
        original_message: The original message text that was filtered.
        channel_id: Optional channel identifier.
        timestamp: Unix timestamp of the message (for TTL validation).
    """
    import time as _time

    ts = timestamp or _time.time()
    preview = original_message[:100] + ("..." if len(original_message) > 100 else "")

    button = _build_callback_button(
        "⚡ 强制处理",
        "force_process",
        channel_id=channel_id,
        button_type="primary",
        extra_value={
            "original_message": original_message[:2000],
            "timestamp": str(ts),
        },
    )

    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": f"💬 **消息预览：** {preview}",
        },
        {
            "tag": "markdown",
            "content": "ℹ️ 此消息被判定为非技术内容，未路由到 Agent。\n如需强制处理，请点击下方按钮或在消息前加 `!` 前缀重新发送。",
        },
    ]
    elements.extend(build_responsive_layout([button]))

    return _build_card_wrapper(
        header_title="💡 消息未处理",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )


# ---------------------------------------------------------------------------
# Routing enhancement cards — queue & transfer
# ---------------------------------------------------------------------------


def build_queue_waiting_card(
    agent: AgentIdentity,
    *,
    channel_id: str = "",
    position: int = 1,
    current_status: str = "running",
) -> dict:
    """Build a card notifying the user that their request is queued.

    Shown when the @mentioned agent is BUSY (non-IDLE status).
    """
    status_zh = _STATUS_LABEL_ZH.get(current_status, current_status)
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": (
                f"⏳ **{agent.emoji} {agent.name}** 当前状态: **{status_zh}**\n"
                f"您的请求已排入队列（位置 #{position}），Agent 空闲后将自动处理。"
            ),
        },
    ]
    # AC26: Use build_responsive_layout for mobile-friendly button rendering
    elements.extend(
        build_responsive_layout([
            _build_callback_button(
                "🔄 查看可用 Agent",
                "slock_show_idle_agents",
                channel_id=channel_id,
                button_type="default",
            ),
            _build_callback_button(
                "❌ 取消排队",
                "slock_cancel_queue",
                channel_id=channel_id,
                button_type="danger",
                extra_value={"agent_id": agent.agent_id},
            ),
        ])
    )

    return _build_card_wrapper(
        header_title="📋 请求已排队",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )


def build_queue_full_card(
    agent: AgentIdentity,
    *,
    channel_id: str = "",
    original_message: str = "",
) -> dict:
    """Build an error card when the mention queue for an agent is full.

    Provides retry and force-interrupt options.
    """
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": (
                f"🚫 **{agent.emoji} {agent.name}** 的消息队列已满（最多 8 条排队）。\n"
                f"您的消息未能入队：\n> {original_message[:200]}"
            ),
        },
    ]
    elements.extend(
        build_responsive_layout([
            _build_callback_button(
                "🔄 重试",
                "slock_queue_retry",
                channel_id=channel_id,
                button_type="primary",
                extra_value={"agent_id": agent.agent_id, "original_message": original_message[:2000]},
            ),
            _build_callback_button(
                "⚡ 强制介入",
                "slock_force_interrupt",
                channel_id=channel_id,
                button_type="danger",
                extra_value={"agent_id": agent.agent_id, "original_message": original_message[:2000]},
            ),
        ])
    )

    return _build_card_wrapper(
        header_title="🚫 队列已满",
        header_template="red",
        elements=elements,
        mobile_optimize=False,
    )


def build_transfer_suggestion_card(
    busy_agent: AgentIdentity,
    idle_agent: AgentIdentity,
    *,
    channel_id: str = "",
    original_message: str = "",
) -> dict:
    """Build a card suggesting transfer from busy agent to an idle same-role agent.

    Shown when @mentioned agent is busy but another agent with the same role is idle.
    """
    preview = original_message[:80] + ("..." if len(original_message) > 80 else "")
    elements: list[dict] = [
        {
            "tag": "markdown",
            "content": (
                f"🔀 **{busy_agent.emoji} {busy_agent.name}** 当前忙碌\n"
                f"发现同角色空闲 Agent: **{idle_agent.emoji} {idle_agent.name}**\n"
                f"是否将此请求转交？"
            ),
        },
    ]
    if preview:
        elements.append(
            {"tag": "markdown", "content": f"> {preview}"}
        )
    # AC26: Use build_responsive_layout for mobile-friendly button rendering
    elements.extend(
        build_responsive_layout([
            _build_callback_button(
                _truncate_dynamic_label(f"✅ 转交给 {idle_agent.name}", max_len=16),
                "slock_transfer_accept",
                channel_id=channel_id,
                button_type="primary",
                extra_value={
                    "from_agent_id": busy_agent.agent_id,
                    "to_agent_id": idle_agent.agent_id,
                    "original_message": original_message[:2000],
                },
            ),
            _build_callback_button(
                "⏳ 继续等待",
                "slock_queue_keep_waiting",
                channel_id=channel_id,
                button_type="default",
                extra_value={"agent_id": busy_agent.agent_id},
            ),
        ])
    )

    return _build_card_wrapper(
        header_title="🔀 转交建议",
        header_template="indigo",
        elements=elements,
        mobile_optimize=False,
    )


def build_conflict_escalation_card(
    *,
    agent_name: str,
    conflict_details: str,
    conclusion: str,
    key_knowledge: str,
    channel_id: str = "",
    thread_id: str = "",
) -> dict:
    """Build a conflict escalation card for human confirmation.

    Shows the detected conflict, conclusion preview, key knowledge preview,
    and provides accept/reject buttons for human decision.

    Args:
        agent_name: Name of the agent whose memory has the conflict.
        conflict_details: Details about the detected conflict.
        conclusion: The discussion conclusion text.
        key_knowledge: The existing key knowledge that conflicts.
        channel_id: Channel ID for action routing.
        thread_id: Discussion thread ID.

    Returns:
        Feishu interactive card dict.
    """
    elements: list[dict] = []

    # Conflict details
    elements.append({
        "tag": "markdown",
        "content": f"**⚠️ 检测到知识冲突**\n\n**Agent:** {agent_name}\n\n**冲突详情:**\n{conflict_details[:300]}",
    })

    elements.append({"tag": "hr"})

    # Conclusion preview (collapsible)
    conclusion_preview = conclusion[:400]
    if len(conclusion) > 400:
        conclusion_preview += "..."

    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "plain_text", "content": "📝 讨论结论"}},
        "vertical_spacing": "8px",
        "elements": [
            {"tag": "markdown", "content": redact_sensitive(conclusion_preview)},
        ],
    })

    # Key Knowledge preview (collapsible)
    kk_preview = key_knowledge[:400]
    if len(key_knowledge) > 400:
        kk_preview += "..."

    elements.append({
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {"title": {"tag": "plain_text", "content": "🔑 关键知识 (Key Knowledge)"}},
        "vertical_spacing": "8px",
        "elements": [
            {"tag": "markdown", "content": redact_sensitive(kk_preview)},
        ],
    })

    elements.append({"tag": "hr"})

    # Action buttons: Accept (override) / Reject (keep)
    accept_value = {
        "action": "slock_conflict_resolve",
        "decision": "accept",
        "thread_id": thread_id,
        "agent_name": agent_name,
        "channel_id": channel_id,
    }
    reject_value = {
        "action": "slock_conflict_resolve",
        "decision": "reject",
        "thread_id": thread_id,
        "agent_name": agent_name,
        "channel_id": channel_id,
    }

    elements.extend(
        build_responsive_layout(
            [
                _build_callback_button(
                    "✅ 接受结论（覆盖）",
                    "slock_conflict_resolve",
                    channel_id=channel_id,
                    button_type="primary",
                    extra_value=accept_value,
                ),
                _build_callback_button(
                    "❌ 拒绝结论（保留）",
                    "slock_conflict_resolve",
                    channel_id=channel_id,
                    button_type="danger",
                    extra_value=reject_value,
                ),
            ]
        )
    )

    # Footer hint
    elements.append({
        "tag": "markdown",
        "content": (
            "💡 **说明:**\n"
            "- 「接受结论」将用新结论覆盖 Agent 的 Key Knowledge\n"
            "- 「拒绝结论」将保留现有 Key Knowledge，跳过本次同步"
        ),
        "text_size": "notation",
    })

    return _build_card_wrapper(
        header_title="⚠️ 知识冲突确认",
        header_template="orange",
        elements=elements,
        mobile_optimize=False,
    )
