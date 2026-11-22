"""The single typed product and command catalog for GhostAP.

Completion labels describe support expectations; they do not gate execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..mode.backends import ALL_BACKEND_METAS


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
    slash_discoverable: bool = True
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
    slash_discoverable: bool = True,
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
        slash_discoverable=slash_discoverable,
        enters_programming_mode=enters_programming_mode,
        programming_mode_id=programming_mode_id,
        protects_from_auto_activation=protects_from_auto_activation,
    )


_EXIT_COMMAND_ALIASES: tuple[str, ...] = (
    "/quit",
    *(
        alias
        for spec in ALL_BACKEND_METAS
        if spec.legacy_exit_mode in {"dedicated_intent", "shared_exit_alias"}
        for alias in (f"/end_{spec.mode_key}", f"/exit_{spec.mode_key}")
    ),
)


def _programming_entry_actions() -> tuple[ProductAction, ...]:
    return tuple(
        _action(
            spec.slash_command,
            (
                f"进入 {spec.display_name} 编程模式"
                if spec.visible_in_tools_list
                else f"进入 {spec.display_name} 桥接模式"
            ),
            aliases=(spec.enter_alias_command,),
            enters_programming_mode=True,
            programming_mode_id=spec.mode_key,
            protects_from_auto_activation=True,
            slash_discoverable=spec.visible_in_tools_list,
        )
        for spec in ALL_BACKEND_METAS
    )


def _info_actions() -> tuple[ProductAction, ...]:
    return tuple(
        _action(spec.info_command, f"查看 {spec.display_name} 会话与模型信息")
        for spec in ALL_BACKEND_METAS
        if spec.supports_info_command
    )


# This is the pre-catalog main Slash registration, moved here verbatim.  The
# Feishu registration module only projects this tuple; it owns no command list.
_RAW_PUBLIC_ACTIONS: tuple[ProductAction, ...] = (
    _action("/help", "查看 GhostAP 完整帮助"),
    _action("/menu", "打开 GhostAP 快捷操作菜单"),
    _action("/tools", "查看可用的 AI 编程工具"),
    _action("/tools_status", "查看 AI 编程工具状态"),
    *_programming_entry_actions(),
    _action("/acp", "查看或设置项目 ACP 工具", "/acp [工具]", protects_from_auto_activation=True),
    _action("/cmds", "查看当前工具原生命令与 Skills", "/cmds [工具]", aliases=("/commands", "/tool_cmds"), protects_from_auto_activation=True),
    _action("/info", "查看当前会话或工作区详情", protects_from_auto_activation=True, slash_discoverable=False),
    _action("/model", "自适应查看或设置当前编程工具模型", "/model [模型名|default]"),
    _action("/effort", "查看或设置当前模型的思考 Effort", "/effort [档位|planner executor|default]"),
    _action("/stop", "立即停止当前运行中的任务（保持当前模式）", protects_from_auto_activation=True),
    _action("/clear", "清空当前会话上下文（保留当前工具与模型）", aliases=("/reset",), protects_from_auto_activation=True),
    _action("/sh", "在当前项目目录直接执行 Shell 命令", "/sh <命令>", aliases=("/shell",), protects_from_auto_activation=True, slash_discoverable=False),
    _action("/ask", "向指定工具发起一次性零副作用咨询", "/ask <工具> <问题>", protects_from_auto_activation=True, slash_discoverable=False),
    _action("/exit", "退出当前编程模式", aliases=_EXIT_COMMAND_ALIASES),
    _action("/btw", "在当前编程会话中提出旁路问题", "/btw <问题>"),
    _action("/coco_status", "查看 Coco 会话状态"),
    *_info_actions(),
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
    _action("/spec_config", "查看 Spec Engine 配置", protects_from_auto_activation=True),
    _action(
        "/spec_export",
        "导出 Spec Engine 报告",
        protects_from_auto_activation=True,
        slash_discoverable=False,
    ),
    _action("/spec_save", "立即保存 Spec Engine 状态", protects_from_auto_activation=True),
    _action("/spec_guide", "补充 Spec Engine 引导", "/spec_guide <引导>", protects_from_auto_activation=True),
    _action("/stop_spec", "停止 Spec Engine 任务", protects_from_auto_activation=True),
    _action("/wf", "启动多 Agent Workflow", "/wf <需求>", aliases=("/workflow",)),
    _action("/wf_status", "查看当前 Workflow 进度", aliases=("/workflow_status",)),
    _action("/wf_help", "查看 Workflow 使用帮助", aliases=("/workflow_help",)),
    _action("/stop_wf", "停止正在运行的 Workflow", aliases=("/stop_workflow",)),
)

# Commands kept out of the Feishu slash-command auto-completion panel.
# They stay fully resolvable when typed explicitly (and keep working via
# cards/buttons); this only trims the discovery surface down to the ~19
# most-used commands.  The SlashCommandReconciler deletes any panel entries
# whose action leaves the discoverable set, so removing a name here prunes it
# from Feishu automatically on the next startup.
HIDDEN_SLASH_COMMANDS: frozenset[str] = frozenset({
    # tool introspection / low-level ACP
    "/tools", "/tools_status", "/acp", "/effort", "/btw", "/sh", "/ask", "/info",
    "/coco_status",
    *(spec.info_command for spec in ALL_BACKEND_METAS if spec.supports_info_command),
    # project administration (board stays reachable via cards)
    "/new-chat", "/close", "/tasks", "/diff", "/trace",
    "/lock", "/unlock", "/setadmin",
    # Deep Engine auxiliaries
    "/deep_status", "/deep_update", "/stop_deep",
    # Spec Engine auxiliaries
    "/spec_status", "/spec_history", "/spec_metrics", "/spec_config",
    "/spec_export", "/spec_save", "/spec_guide", "/stop_spec",
    # Workflow auxiliaries
    "/wf_status", "/wf_help", "/stop_wf",
    # hidden local CLI bridges — type explicitly, never advertised
    *(spec.slash_command for spec in ALL_BACKEND_METAS if not spec.visible_in_tools_list),
})

PUBLIC_ACTIONS: tuple[ProductAction, ...] = tuple(
    replace(action, slash_discoverable=False)
    if action.command in HIDDEN_SLASH_COMMANDS
    else action
    for action in _RAW_PUBLIC_ACTIONS
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
        tuple(spec.cli_name for spec in ALL_BACKEND_METAS),
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


def get_slash_discoverable_actions() -> tuple[ProductAction, ...]:
    """Public commands whose user journey is complete in Feishu chat."""

    return tuple(action for action in PUBLIC_ACTIONS if action.slash_discoverable)


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
