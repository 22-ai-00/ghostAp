"""Canonical Feishu interactive cards for the autonomous Manager.

All cards follow the pattern: structured JSON templates rendered
with dynamic data. Reuses the project's shared tool/model discovery
from src/workflow_engine/tool_registry (same source as /wf, Deep, Spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..provisioning.hire_port import EmployeeProfileTemplate


@dataclass(frozen=True, slots=True)
class EmployeeRuntimeCardView:
    """Secret-free facade output consumed by cards and handlers."""

    agent_id: str
    name: str
    emoji: str
    role: str
    tool: str
    model: str
    employee_state: str
    bot_state: str
    bot_generation: int
    actor_state: str
    mailbox_depth: int
    can_accept: bool
    identity_version: int
    knowledge_generation: int
    active_assignment_id: str = ""
    active_run_id: str = ""
    last_checkpoint: str = ""
    context_quality: str = "complete"
    context_warnings: tuple[str, ...] = ()
    error_code: str = ""
    completed_contributions: tuple[str, ...] = ()
    recovery_hint: str = ""
    review_item_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_warnings", tuple(self.context_warnings))
        object.__setattr__(
            self,
            "completed_contributions",
            tuple(self.completed_contributions),
        )
        object.__setattr__(self, "review_item_ids", tuple(self.review_item_ids))
        if self.mailbox_depth < 0 or self.bot_generation < 0:
            raise ValueError("runtime counters must be non-negative")


def _runtime_markdown(view: EmployeeRuntimeCardView) -> list[dict[str, Any]]:
    session_temperature = {
        "ready_warm": "warm",
        "busy": "warm",
        "ready_cold": "cold",
    }.get(view.actor_state, "unavailable")
    admission = "可接任务" if view.can_accept else "不可接任务"
    model_label = f"{view.tool} / {view.model or 'default'}"
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": (
                f"{view.emoji} **{view.name}** · `{view.role or 'custom'}`\n"
                f"{model_label} · identity v{view.identity_version}"
            ),
        },
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": (
                f"**飞书连接**　Bot {view.bot_state.upper()} · generation {view.bot_generation}\n"
                f"**模型运行时**　Agent {view.actor_state.upper()} · session {session_temperature}\n"
                f"**接单资格**　{admission} · mailbox {view.mailbox_depth}\n"
                f"**持久知识**　knowledge generation {view.knowledge_generation}"
            ),
        },
    ]
    if view.active_assignment_id or view.active_run_id or view.last_checkpoint:
        parts = ["**当前 Assignment**"]
        if view.active_run_id:
            parts.append(f"Run: `{view.active_run_id}`")
        if view.active_assignment_id:
            parts.append(f"Assignment: `{view.active_assignment_id}`")
        if view.last_checkpoint:
            parts.append(f"Last checkpoint: `{view.last_checkpoint}`")
        elements.extend(({"tag": "hr"}, {"tag": "markdown", "content": "\n".join(parts)}))
    if view.context_quality != "complete":
        warning_text = "、".join(view.context_warnings) or "部分来源暂不可用"
        elements.extend(
            (
                {"tag": "hr"},
                {
                    "tag": "markdown",
                    "content": (
                        "**⚠️ 上下文部分可用（降级说明）**\n"
                        f"任务可继续，但结果会保留 partial 标记：{warning_text}"
                    ),
                },
            )
        )
    if view.error_code:
        contributions = "\n".join(f"- {item}" for item in view.completed_contributions)
        detail = (
            f"**需要人工处理** · `{view.error_code}`\n"
            f"Run: `{view.active_run_id or 'unknown'}` · "
            f"Assignment: `{view.active_assignment_id or 'unknown'}`"
        )
        if contributions:
            detail += f"\n**已完成贡献**\n{contributions}"
        if view.recovery_hint:
            detail += f"\n**恢复动作**　{view.recovery_hint}"
        elements.extend(({"tag": "hr"}, {"tag": "markdown", "content": detail}))
    return elements


def build_employee_runtime_status_card(
    view: EmployeeRuntimeCardView,
    *,
    admin: bool = False,
) -> dict[str, Any]:
    """Render one employee runtime without conflating transport and model state."""

    elements = _runtime_markdown(view)
    if admin:
        actions = [
            ("回收模型会话", "employee_runtime_recycle_session", ""),
            ("重建 Workspace", "employee_runtime_rebuild_workspace", ""),
            ("检查 Knowledge", "employee_runtime_lint_knowledge", ""),
        ]
        if view.review_item_ids:
            actions.append(
                (
                    f"重试 Review Item ({len(view.review_item_ids)})",
                    "employee_runtime_retry_review",
                    view.review_item_ids[0],
                )
            )
        elements.extend(
            (
                {"tag": "hr"},
                {"tag": "markdown", "content": "**管理员恢复动作**"},
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": label},
                            "type": "primary" if index == 0 else "default",
                            "value": {
                                "action": action,
                                "agent_id": view.agent_id,
                                **({"review_id": review_id} if review_id else {}),
                            },
                        }
                        for index, (label, action, review_id) in enumerate(actions)
                    ],
                },
            )
        )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"员工运行时 · {view.name}"},
            "template": "orange" if view.error_code else "blue",
        },
        "body": {"elements": elements},
    }


def build_employee_runtime_overview_card(
    employees: tuple[EmployeeRuntimeCardView, ...],
) -> dict[str, Any]:
    """Render the complete, secret-free runtime state of a tenant's employees."""

    elements: list[dict[str, Any]] = []
    for index, view in enumerate(employees):
        if index:
            elements.append({"tag": "hr"})
        elements.extend(_runtime_markdown(view))
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"员工运行时总览（{len(employees)}人）",
            },
            "template": "orange" if any(view.error_code for view in employees) else "blue",
        },
        "body": {"elements": elements},
    }


def build_employee_roster_card(
    employees: tuple[EmployeeRuntimeCardView, ...],
    *,
    archived_count: int = 0,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    for view in employees:
        admission = "可接任务" if view.can_accept else "不可接任务"
        content = (
            f"{view.emoji} **{view.name}** · `{view.tool}/{view.model or 'default'}`\n"
            f"Bot {view.bot_state.upper()} / Agent {view.actor_state.upper()} · {admission}"
        )
        if view.active_assignment_id:
            content += f"\n当前 Assignment: `{view.active_assignment_id}`"
        elements.append({"tag": "markdown", "content": content})
        elements.append({"tag": "hr"})
    if elements:
        elements.pop()
    if archived_count:
        elements.append(
            {"tag": "note", "elements": [{"tag": "plain_text", "content": f"历史归档 {archived_count} 人"}]}
        )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"员工花名册（{len(employees)}人）"},
            "template": "blue",
        },
        "body": {"elements": elements},
    }


def build_employee_profile_markdown(
    *,
    employee_name: str,
    tool: str,
    model: str,
    profile: EmployeeProfileTemplate,
) -> str:
    """Render the bounded profile shown before a visible hire is admitted."""

    model_label = model or "provider default"
    return (
        f"**员工资料预览 · {employee_name}**\n"
        f"职责：`{profile.role}` · 工具/模型：`{tool}` / `{model_label}`\n"
        f"工作风格：{profile.persona}\n"
        f"性格：{', '.join(profile.personality_traits)}\n"
        f"能力：{', '.join(profile.capabilities)}\n"
        f"权限策略：`{profile.permission_profile}`（{', '.join(profile.permissions)}）\n\n"
        "资料来自版本化模板，不接受任意 system prompt；名字和性格不会隐式扩权。"
    )


def _discover_tools() -> list[str]:
    """Get available tools using the same discovery as Workflow/Deep/Spec."""
    try:
        from ...workflow_engine.tool_registry import get_available_tools
        tools = get_available_tools(require_available=True)
        if tools:
            return list(tools.keys())
    except Exception:
        pass
    try:
        from ...acp.providers import get_providers
        providers = get_providers()
        if providers:
            return list(providers.keys())
    except Exception:
        pass
    return ["coco", "claude", "codex", "aiden", "gemini", "traex"]


def _discover_models_for_tool(tool_name: str) -> list[str]:
    """Get available models for a tool using ACP provider model discovery."""
    try:
        from ...acp.providers import get_providers
        providers = get_providers()
        provider = providers.get(tool_name)
        if provider and hasattr(provider, "get_models"):
            models = provider.get_models()
            if models:
                return [m.name if hasattr(m, "name") else str(m) for m in models]
        if provider and hasattr(provider, "default_model"):
            dm = provider.default_model
            if dm:
                return [dm]
    except Exception:
        pass
    return []


def build_employee_creation_card(
    *,
    available_roles: list[str] | None = None,
    available_tools: list[str] | None = None,
    available_models: list[str] | None = None,
) -> dict[str, Any]:
    """Build interactive card for creating a new employee via Feishu chat.

    Tool/model lists are discovered from the same ACP provider registry
    that Deep, Spec, and Workflow use. Falls back to static
    defaults only when discovery fails.
    """
    roles = available_roles or ["coder", "reviewer", "planner", "tester", "researcher"]
    tools = available_tools or _discover_tools()
    models = available_models or _get_all_models(tools)

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Create New Employee"},
            "template": "blue",
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "Configure a new autonomous employee for your team:",
                },
            },
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        "**Work Style:** version-controlled role template\n"
                        "**Capabilities:** bounded allowlist\n"
                        "**Permission Profile:** administrator-selected least privilege\n"
                        "Arbitrary system prompt input is not accepted."
                    ),
                },
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "select_static",
                        "placeholder": {"tag": "plain_text", "content": "Select Role"},
                        "options": [
                            {"text": {"tag": "plain_text", "content": r}, "value": r}
                            for r in roles
                        ],
                        "value": {"key": "employee_role"},
                    },
                ],
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "select_static",
                        "placeholder": {"tag": "plain_text", "content": "Select Tool"},
                        "options": [
                            {"text": {"tag": "plain_text", "content": t}, "value": t}
                            for t in tools
                        ],
                        "value": {"key": "employee_tool"},
                    },
                ],
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "select_static",
                        "placeholder": {"tag": "plain_text", "content": "Select Model (or use tool default)"},
                        "options": [
                            {"text": {"tag": "plain_text", "content": "Auto (tool default)"}, "value": "__auto__"},
                            *[
                                {"text": {"tag": "plain_text", "content": m}, "value": m}
                                for m in models
                            ],
                        ],
                        "value": {"key": "employee_model"},
                    },
                ],
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Create Employee"},
                        "type": "primary",
                        "value": {"action": "create_employee"},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Cancel"},
                        "type": "default",
                        "value": {"action": "cancel"},
                    },
                ],
            },
        ],
    }


def _get_all_models(tools: list[str]) -> list[str]:
    """Collect unique models from all discovered tools."""
    seen: set[str] = set()
    models: list[str] = []
    for tool in tools:
        for m in _discover_models_for_tool(tool):
            if m not in seen:
                seen.add(m)
                models.append(m)
    return models


def build_employee_created_card(
    *,
    employee_id: str,
    name: str,
    role: str,
    tool: str,
    model: str,
    worker_type: str,
) -> dict[str, Any]:
    """Build confirmation card after employee creation."""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Employee Created"},
            "template": "green",
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**ID:** {employee_id}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Name:** {name}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Role:** {role}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Tool:** {tool}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Model:** {model}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Type:** {worker_type}"}},
                ],
            },
            {
                "tag": "note",
                "elements": [
                    {"tag": "plain_text", "content": "Employee is now active and ready to accept work."},
                ],
            },
        ],
    }


def build_goal_progress_card(
    *,
    goal_id: str,
    description: str,
    state: str,
    run_id: str = "",
    step_progress: str = "",
) -> dict[str, Any]:
    """Build progress card for goal execution."""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"Goal: {description[:40]}"},
            "template": "wathet" if state == "executing" else "blue",
        },
        "elements": [
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Goal:** {goal_id}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**State:** {state}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Run:** {run_id or 'N/A'}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Progress:** {step_progress or 'Starting...'}"}},
                ],
            },
        ],
    }


def build_approval_card(
    *,
    approval_id: str,
    description: str,
    risk_level: str,
    effect_summary: str,
) -> dict[str, Any]:
    """Build approval request card."""
    template = "red" if risk_level in ("r3", "r4") else "orange"
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Approval Required"},
            "template": template,
        },
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**Action:** {description}"},
            },
            {
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Risk:** {risk_level.upper()}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**Effect:** {effect_summary}"}},
                ],
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Approve"},
                        "type": "primary",
                        "value": {"action": "approve", "approval_id": approval_id},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Reject"},
                        "type": "danger",
                        "value": {"action": "reject", "approval_id": approval_id},
                    },
                ],
            },
        ],
    }
