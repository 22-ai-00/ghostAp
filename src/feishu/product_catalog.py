"""The single typed product and command catalog for GhostAP.

Completion labels describe support expectations.  They never introduce an
Owner access, rollout, allowlist, approval, sandbox, or release-state gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class CompletionLabel(str, Enum):
    MATURE = "mature"
    DEVELOPING = "developing"
    NOT_IMPLEMENTED = "not_implemented"


class RuntimeHealth(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ExecutionLane(str, Enum):
    DIRECT = "direct"
    DEEP = "deep"
    SPEC = "spec"
    WORKFLOW = "workflow"


class ProductRole(str, Enum):
    OWNER = "owner"


class ProductScope(str, Enum):
    OWNER = "owner"
    PROJECT = "project"
    TEAM = "team"


class CompatibilityBehavior(str, Enum):
    PASSTHROUGH = "passthrough"
    REWRITE = "rewrite"
    RETIRED_MESSAGE = "retired_message"


@dataclass(frozen=True, slots=True)
class ProductAction:
    """One canonical public command and its compatibility metadata."""

    action_id: str
    command: str
    label: str
    description: str
    usage: str = ""
    aliases: tuple[str, ...] = ()
    roles: frozenset[ProductRole] = frozenset({ProductRole.OWNER})
    scopes: frozenset[ProductScope] = frozenset({ProductScope.OWNER})
    owner_accessible: bool = True
    compatibility: CompatibilityBehavior = CompatibilityBehavior.PASSTHROUGH
    public: bool = True
    enters_programming_mode: bool = False
    programming_mode_id: str | None = None
    protects_from_auto_activation: bool = False
    lane: ExecutionLane | None = None
    completion: CompletionLabel | None = None
    runtime_health: RuntimeHealth | None = None
    blocking_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProductCommand:
    action: ProductAction
    invoked_as: str
    arguments: str
    rewritten_command: str | None = None


_OWNER_SCOPES = frozenset({ProductScope.OWNER, ProductScope.PROJECT, ProductScope.TEAM})


def _action(
    command: str,
    description: str,
    usage: str = "",
    *,
    aliases: tuple[str, ...] = (),
    compatibility: CompatibilityBehavior = CompatibilityBehavior.PASSTHROUGH,
    enters_programming_mode: bool = False,
    programming_mode_id: str | None = None,
    protects_from_auto_activation: bool = False,
) -> ProductAction:
    return ProductAction(
        action_id=command.removeprefix("/").replace("_", "-"),
        command=command,
        label=command,
        description=description,
        usage=usage,
        aliases=aliases,
        scopes=_OWNER_SCOPES,
        compatibility=compatibility,
        enters_programming_mode=enters_programming_mode,
        programming_mode_id=programming_mode_id,
        protects_from_auto_activation=protects_from_auto_activation,
    )


# This is the pre-catalog main Slash registration, moved here verbatim.  The
# Feishu registration module only projects this tuple; it owns no command list.
PUBLIC_ACTIONS: tuple[ProductAction, ...] = (
    _action("/help", "查看 GhostAP 完整帮助"),
    _action("/menu", "打开 GhostAP 快捷操作菜单"),
    _action("/tools", "查看可用的 AI 编程工具"),
    _action("/tools_status", "查看 AI 编程工具状态"),
    _action("/coco", "进入 Coco 编程模式", aliases=("/enter_coco",), enters_programming_mode=True, programming_mode_id="coco", protects_from_auto_activation=True),
    _action("/claude", "进入 Claude 编程模式", aliases=("/enter_claude",), enters_programming_mode=True, programming_mode_id="claude", protects_from_auto_activation=True),
    _action("/aiden", "进入 Aiden 编程模式", aliases=("/enter_aiden",), enters_programming_mode=True, programming_mode_id="aiden", protects_from_auto_activation=True),
    _action("/codex", "进入 Codex 编程模式", aliases=("/enter_codex",), enters_programming_mode=True, programming_mode_id="codex", protects_from_auto_activation=True),
    _action("/gemini", "进入 Gemini 编程模式", aliases=("/enter_gemini",), enters_programming_mode=True, programming_mode_id="gemini", protects_from_auto_activation=True),
    _action("/traex", "进入 Traex 编程模式", aliases=("/enter_traex",), enters_programming_mode=True, programming_mode_id="traex", protects_from_auto_activation=True),
    _action("/acp", "查看或设置项目 ACP 工具", "/acp [工具]", protects_from_auto_activation=True),
    _action("/model", "查看或设置项目模型", "/model [模型名|default]"),
    _action("/exit", "退出当前编程模式", aliases=("/quit", "/end_coco", "/exit_coco", "/end_claude", "/exit_claude", "/end_aiden", "/exit_aiden", "/end_codex", "/exit_codex", "/end_gemini", "/exit_gemini", "/end_traex", "/exit_traex")),
    _action("/btw", "在当前编程会话中提出旁路问题", "/btw <问题>"),
    _action("/coco_status", "查看 Coco 会话状态"),
    _action("/coco_info", "查看 Coco 会话与模型信息"),
    _action("/claude_info", "查看 Claude 会话与模型信息"),
    _action("/aiden_info", "查看 Aiden 会话与模型信息"),
    _action("/codex_info", "查看 Codex 会话与模型信息"),
    _action("/gemini_info", "查看 Gemini 会话与模型信息"),
    _action("/traex_info", "查看 Traex 会话与模型信息"),
    _action("/projects", "查看项目面板"),
    _action("/new", "创建项目", "/new <名称> [目录]"),
    _action("/new-chat", "创建或绑定项目群", "/new-chat <名称> [后缀] [目录]"),
    _action("/switch", "切换项目", "/switch <名称>"),
    _action("/close", "关闭项目", "/close <名称>"),
    _action("/status", "查看项目、模式、锁与任务状态", "/status [任务ID]"),
    _action("/tasks", "查看任务看板", "/tasks [筛选条件]"),
    _action("/diff", "查看最近版本变更", "/diff [范围]"),
    _action("/trace", "查看消息处理链路", "/trace <消息ID>"),
    _action("/lock", "锁定当前聊天的任务执行"),
    _action("/unlock", "解除当前聊天锁定"),
    _action("/setadmin", "设置或替换 GhostAP 管理员", "/setadmin [用户ID]"),
    _action("/deep", "启动 Deep Engine 复杂任务", "/deep <需求>", protects_from_auto_activation=True),
    _action("/deep_status", "查看 Deep Engine 任务进度", protects_from_auto_activation=True),
    _action("/deep_update", "补充 Deep Engine 任务约束", "/deep_update <补充说明>", protects_from_auto_activation=True),
    _action("/stop_deep", "停止 Deep Engine 任务", protects_from_auto_activation=True),
    _action("/spec", "启动 Spec Engine 结构化开发", "/spec <需求>", protects_from_auto_activation=True),
    _action("/spec_status", "查看 Spec Engine 任务进度", protects_from_auto_activation=True),
    _action("/spec_history", "查看 Spec Engine 历史", protects_from_auto_activation=True),
    _action("/spec_metrics", "查看 Spec Engine 目标达成度", protects_from_auto_activation=True),
    _action("/spec_config", "查看或调整 Spec Engine 配置", protects_from_auto_activation=True),
    _action("/spec_export", "导出 Spec Engine 报告", protects_from_auto_activation=True),
    _action("/spec_save", "立即保存 Spec Engine 状态", protects_from_auto_activation=True),
    _action("/spec_guide", "补充 Spec Engine 引导", "/spec_guide <引导>", protects_from_auto_activation=True),
    _action("/stop_spec", "停止 Spec Engine 任务", protects_from_auto_activation=True),
    _action("/wf", "启动多 Agent Workflow", "/wf <需求>", aliases=("/workflow",)),
    _action("/wf_status", "查看当前 Workflow 进度", aliases=("/workflow_status",)),
    _action("/wf_help", "查看 Workflow 使用帮助", aliases=("/workflow_help",)),
    _action("/stop_wf", "停止正在运行的 Workflow", aliases=("/stop_workflow",)),
    _action("/hire", "雇佣持久数字员工", "/hire <名字> [选项]", aliases=("/h",)),
    _action("/fire", "退役持久数字员工", "/fire <名字>"),
    _action("/employees", "查看在职数字员工", aliases=("/roster",)),
    _action("/history", "查看数字员工执行历史", "/history [员工名]"),
    _action("/employee-memory", "查看数字员工记忆摘要", "/employee-memory <员工名>"),
)


COMPATIBILITY_ACTIONS: tuple[ProductAction, ...] = (
    ProductAction(
        action_id="retired-autonomous-manager",
        command="/goal",
        label="retired autonomous manager",
        description="Retired command migration response.",
        aliases=("/goals", "/run", "/runs", "/approve", "/approvals", "/decisions"),
        scopes=_OWNER_SCOPES,
        compatibility=CompatibilityBehavior.RETIRED_MESSAGE,
        public=False,
    ),
)


_PUBLIC_BY_COMMAND = {action.command: action for action in PUBLIC_ACTIONS}
_PUBLIC_BY_ACTION_ID = {action.action_id: action for action in PUBLIC_ACTIONS}
_COMPATIBILITY_BY_COMMAND = {
    token: action
    for action in PUBLIC_ACTIONS
    for token in action.aliases
}
_COMPATIBILITY_BY_COMMAND.update(
    {
        token: action
        for action in COMPATIBILITY_ACTIONS
        for token in (action.command, *action.aliases)
    }
)
COMPATIBILITY_TOKENS = frozenset(
    token
    for action in COMPATIBILITY_ACTIONS
    for token in (action.command, *action.aliases)
)


@dataclass(frozen=True, slots=True)
class ExecutionLaneDescriptor:
    """Support metadata whose command fields are projected from public actions."""

    lane: ExecutionLane
    primary_action_id: str
    label: str
    completion: CompletionLabel
    runtime_health: RuntimeHealth
    additional_action_ids: tuple[str, ...] = ()
    compatibility_alias_action_ids: tuple[str, ...] = ()
    blocking_reason: str | None = None


_EXECUTION_LANE_DESCRIPTORS: tuple[ExecutionLaneDescriptor, ...] = (
    ExecutionLaneDescriptor(
        ExecutionLane.DIRECT,
        "acp",
        "Direct 编程",
        CompletionLabel.MATURE,
        RuntimeHealth.AVAILABLE,
        ("coco", "claude", "aiden", "codex", "gemini", "traex"),
    ),
    ExecutionLaneDescriptor(ExecutionLane.DEEP, "deep", "Deep", CompletionLabel.MATURE, RuntimeHealth.AVAILABLE),
    ExecutionLaneDescriptor(ExecutionLane.SPEC, "spec", "Spec", CompletionLabel.MATURE, RuntimeHealth.AVAILABLE),
    ExecutionLaneDescriptor(ExecutionLane.WORKFLOW, "wf", "Workflow", CompletionLabel.DEVELOPING, RuntimeHealth.AVAILABLE, compatibility_alias_action_ids=("wf",), blocking_reason="冻结 IR v2 与耐久执行端口尚未完成；当前 Workflow 保持既有 RunSpec 与 reviewer 合同。"),
)


def _project_execution_action(descriptor: ExecutionLaneDescriptor) -> ProductAction:
    primary = _PUBLIC_BY_ACTION_ID[descriptor.primary_action_id]
    aliases = tuple(
        _PUBLIC_BY_ACTION_ID[action_id].command
        for action_id in descriptor.additional_action_ids
    ) + tuple(
        alias
        for action_id in descriptor.compatibility_alias_action_ids
        for alias in _PUBLIC_BY_ACTION_ID[action_id].aliases
    )
    return replace(
        primary,
        action_id=f"lane-{descriptor.lane.value}",
        label=descriptor.label,
        aliases=aliases,
        lane=descriptor.lane,
        completion=descriptor.completion,
        runtime_health=descriptor.runtime_health,
        blocking_reason=descriptor.blocking_reason,
    )


EXECUTION_ACTIONS = tuple(
    _project_execution_action(descriptor)
    for descriptor in _EXECUTION_LANE_DESCRIPTORS
)
_LANE_BY_NAME = {action.lane: action for action in EXECUTION_ACTIONS}


def get_public_actions() -> tuple[ProductAction, ...]:
    """All currently implemented Owner-visible commands, never a rollout filter."""

    return PUBLIC_ACTIONS


def get_compatibility_actions() -> tuple[ProductAction, ...]:
    """Aliases and retired spellings retained for deterministic compatibility."""

    return COMPATIBILITY_ACTIONS + tuple(action for action in PUBLIC_ACTIONS if action.aliases)


def retired_command_tokens() -> frozenset[str]:
    """All retired command spellings handled by the compatibility responder."""

    return frozenset(
        token
        for action in COMPATIBILITY_ACTIONS
        if action.compatibility is CompatibilityBehavior.RETIRED_MESSAGE
        for token in (action.command, *action.aliases)
    )


def resolve_command(token: str, arguments: str = "") -> ResolvedProductCommand | None:
    """Resolve a public, compatibility, or retired command without side effects."""

    invoked_as = (token or "").strip().lower()
    action = _PUBLIC_BY_COMMAND.get(invoked_as) or _COMPATIBILITY_BY_COMMAND.get(invoked_as)
    if action is None:
        return None
    rewritten = action.command if (
        invoked_as != action.command
        and action.compatibility is CompatibilityBehavior.REWRITE
    ) else None
    return ResolvedProductCommand(action, invoked_as, arguments, rewritten)


def is_programming_entry_command(text: str) -> bool:
    token = (text or "").strip().split(maxsplit=1)[0].lower() if text else ""
    resolved = resolve_command(token)
    return bool(resolved and resolved.action.enters_programming_mode)


def is_same_programming_mode_entry(mode_id: str, text: str) -> bool:
    """Whether *text* re-enters the programming mode already active in a topic."""

    token = (text or "").strip().split(maxsplit=1)[0].lower() if text else ""
    resolved = resolve_command(token)
    return bool(resolved and resolved.action.programming_mode_id == mode_id)


def get_execution_action(lane: ExecutionLane) -> ProductAction:
    return _LANE_BY_NAME[lane]


def get_owner_actions() -> tuple[ProductAction, ...]:
    """The Owner menu's lane projection; completion is explanatory only."""

    return EXECUTION_ACTIONS


def format_owner_execution_lane_summary() -> str:
    lines = ["**执行通道（完成度不影响 Owner 访问）**"]
    for action in get_owner_actions():
        aliases = " · ".join((action.command, *action.aliases))
        lines.append(f"- **{action.label}** `{aliases}` · {action.completion.value} · {action.runtime_health.value}")
        if action.blocking_reason:
            lines.append(f"  当前缺口：{action.blocking_reason}")
    return "\n".join(lines)


def is_explicit_protected_command(text: str) -> bool:
    token = (text or "").strip().split(maxsplit=1)[0].lower() if text else ""
    resolved = resolve_command(token)
    return bool(resolved and resolved.action.protects_from_auto_activation)
