"""Workflow Engine handler — /wf, /workflow, /stop_wf, /wf_status, /wf_help commands."""

from __future__ import annotations

import json
import logging
import threading
import time  # noqa: F401
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ...card.text_stream import append_stream_text
from ...utils.text import generate_task_id  # noqa: F401
from ..emoji import EmojiReaction
from ..slash_command_parser import SlashCommandParser
from .engine_base import BaseEngineHandler

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ..handler_context import HandlerContext


def __getattr__(name: str) -> Any:
    if name == "workflow_scripts_dir":
        from ...workflow_engine.storage import workflow_scripts_dir

        globals()["workflow_scripts_dir"] = workflow_scripts_dir
        return workflow_scripts_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

logger = logging.getLogger(__name__)

_SCRIPT_GENERATION_MAX_ATTEMPTS = 3
_GENERATION_ACTIVITY_MAX_CHARS = 180
_GENERATION_ACTIVITY_BREAKS = frozenset("。！？.!?；;\n")
_WORKFLOW_STOP_QUIESCENCE_TIMEOUT_S = 5.0
_WORKFLOW_COT_REQUEST_TIMEOUT_S = 5.0
_WORKFLOW_COT_CLOSE_TIMEOUT_S = 2.5


@dataclass(slots=True)
class _WorkflowGenerationActivityBuffer:
    """Coalesce provider text tokens into readable generation activities."""

    pending: str = ""

    def feed(self, text: str) -> tuple[str, ...]:
        if not text:
            return ()
        self.pending = append_stream_text(self.pending, text)
        completed: list[str] = []
        while self.pending:
            boundary = self._first_boundary(self.pending)
            if boundary < 0 and len(self.pending) < _GENERATION_ACTIVITY_MAX_CHARS:
                break
            end = (
                boundary + 1
                if boundary >= 0
                else _GENERATION_ACTIVITY_MAX_CHARS
            )
            segment = self.pending[:end].strip()
            self.pending = self.pending[end:].lstrip()
            if segment:
                completed.append(segment)
        return tuple(completed)

    def flush(self) -> tuple[str, ...]:
        segment = self.pending.strip()
        self.pending = ""
        return (segment,) if segment else ()

    @staticmethod
    def _first_boundary(value: str) -> int:
        return next(
            (
                index
                for index, char in enumerate(value)
                if char in _GENERATION_ACTIVITY_BREAKS
            ),
            -1,
        )


@dataclass(frozen=True, slots=True)
class _WorkflowLifecycleOwner:
    """Linearization token shared by generation, card delivery, and queued start."""

    session_key: str
    initiator_user_id: str = ""
    chat_id: str = ""
    project_id: str = ""
    root_path: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)
    heartbeat_stop_event: threading.Event = field(default_factory=threading.Event)
    delivery_lock: Any = field(default_factory=threading.RLock)
    source_script_path: str | None = None
    execution_script_path: str | None = None
    done_event: threading.Event = field(default_factory=threading.Event)
    claimed_event: threading.Event = field(default_factory=threading.Event)
    worker_started_event: threading.Event = field(
        default_factory=threading.Event,
    )
    worker_exited_event: threading.Event = field(
        default_factory=threading.Event,
    )
    stop_delivery_fenced_event: threading.Event = field(
        default_factory=threading.Event,
    )
    worker_thread_id: int | None = field(
        default=None,
        compare=False,
    )
    active_generation_session: Any = field(
        default=None,
        compare=False,
    )
    card_message_id: str | None = field(
        default=None,
        compare=False,
    )


class _WorkflowGenerationCancelled(RuntimeError):
    """Internal control flow: a superseded generator must not write fallback."""


class _WorkflowGenerationExhausted(RuntimeError):
    """All model paths and the deterministic pool fallback were exhausted."""


def _workflow_pending_statuses():
    """States that own a pending Workflow card/session rather than a runtime run."""
    from ...workflow_engine.models import WorkflowStatus

    return {
        WorkflowStatus.GENERATING_SCRIPT,
    }


from .workflow_config import WorkflowConfigMixin  # noqa: E402
from .workflow_execution import WorkflowExecutionMixin  # noqa: E402
from .workflow_script import WorkflowScriptMixin  # noqa: E402


class WorkflowHandler(
    WorkflowConfigMixin,
    WorkflowExecutionMixin,
    WorkflowScriptMixin,
    BaseEngineHandler,
):
    """Manages the full lifecycle of Workflow Engine tasks.

    Commands:
        /wf <requirement>       — Start a new workflow (AI generates script)
        /stop_wf                — Cancel the running workflow
        /wf_status              — Show current workflow progress
    """

    def __init__(self, ctx: "HandlerContext") -> None:
        super().__init__(ctx)

    # ------------------------------------------------------------------
    # Topic-engine free-text entry point
    # ------------------------------------------------------------------
    def handle_message(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        """Handle free-text messages when auto_enter_mode == 'workflow'.

        Treats the entire text as a workflow requirement and starts generation.
        """
        text_stripped = text.strip()
        if not text_stripped:
            return
        self.start_workflow(message_id, chat_id, text_stripped, project)

    @staticmethod
    def _parse_wf_launch_args(args: str) -> tuple[str | None, str, str | None]:
        """Parse optional leading ``--auto``, ``--quick``, or ``--preset <id>`` from `/wf` args.

        Returns ``(preset_id, remaining_requirement, error_message)``.
        """
        raw = (args or "").strip()
        if not raw:
            return None, "", None
        parts = raw.split(None, 1)
        head = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""
        head_lower = head.lower()
        if head_lower == "--auto":
            return "__auto__", rest, None
        if head_lower == "--quick":
            return "quick", rest, None
        raw_preset: str | None = None
        if head_lower.startswith("--preset="):
            raw_preset = head.split("=", 1)[1].strip()
        elif head_lower == "--preset":
            if not rest:
                return None, "", "请指定预设名称，例如：`/wf --preset collaborative 实现用户认证`"
            subparts = rest.split(None, 1)
            raw_preset = subparts[0].strip()
            rest = subparts[1].strip() if len(subparts) > 1 else ""
        if raw_preset is None:
            return None, raw, None

        cleaned = raw_preset.lower().replace("-", "_")
        if cleaned in {"auto", "custom"}:
            return cleaned, rest, None
        from ...workflow_engine.teamwork import resolve_teamwork_preset

        try:
            preset = resolve_teamwork_preset(cleaned)
        except ValueError:
            return None, rest, f"未知预设 `{raw_preset}`，可选：`collaborative`、`quick`、`review_team`、`deep_collab`、`auto`、`custom`"
        return preset.preset_id, rest, None

    # ------------------------------------------------------------------
    # Command router
    # ------------------------------------------------------------------
    def handle_workflow_command(
        self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None
    ):
        """Route /wf, /workflow, /wf_config, /stop_wf, /wf_status, /wf_help commands."""
        match = SlashCommandParser.parse(text)
        command = match.command if match else ""
        args = match.args if match else ""
        command = {
            "/workflow": "/wf",
            "/workflow_config": "/wf_config",
            "/workflow_status": "/wf_status",
            "/workflow_help": "/wf_help",
            "/stop_workflow": "/stop_wf",
        }.get(command, command)

        if command == "/stop_wf":
            self.stop_workflow(message_id, chat_id, project)
        elif command == "/wf_status":
            self.show_workflow_status(message_id, chat_id, project)
        elif command == "/wf_help":
            self.show_workflow_help(message_id, chat_id=chat_id, project=project)
        elif command == "/wf_config":
            preset_id, req, err = self._parse_wf_launch_args(args)
            if err:
                self._reply_workflow_error(message_id, "invalid_argument", detail=err)
                return
            self.show_workflow_config(
                message_id,
                chat_id,
                project,
                preset_id=("auto" if preset_id == "__auto__" else preset_id),
                requirement=(req or None),
            )
        elif command == "/wf":
            if args:
                preset_id, req, err = self._parse_wf_launch_args(args)
                if err:
                    self._reply_workflow_error(message_id, "invalid_argument", detail=err)
                    return
                if preset_id == "__auto__":
                    if req:
                        self.start_workflow(message_id, chat_id, req, project)
                    else:
                        self._reply_workflow_error(
                            message_id,
                            "invalid_argument",
                            detail="请提供任务需求，例如：`/wf --auto 帮我分析项目优化点`",
                        )
                elif preset_id is not None:
                    if req:
                        self.start_workflow(
                            message_id,
                            chat_id,
                            req,
                            project,
                            preset_id=preset_id,
                        )
                    else:
                        self.show_workflow_config(
                            message_id,
                            chat_id,
                            project,
                            preset_id=preset_id,
                        )
                else:
                    project = self._ensure_project(message_id, chat_id, project)
                    if not project:
                        return
                    root_path = project.root_path if project else self.get_working_dir(chat_id)
                    from ...workflow_engine.models import WorkflowStatus

                    existing = self.ctx.workflow_engine_manager.get(chat_id, root_path)
                    existing_status = (
                        getattr(getattr(existing, "project", None), "status", None)
                        if existing
                        else None
                    )
                    if existing and (
                        existing.is_running is True
                        or existing_status == WorkflowStatus.GENERATING_SCRIPT
                    ):
                        detail = (
                            "当前项目已有 Workflow 正在生成编排脚本。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。"
                            if existing_status == WorkflowStatus.GENERATING_SCRIPT
                            else "当前项目已有 Workflow 任务在执行中。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。"
                        )
                        self._reply_workflow_error(
                            message_id,
                            "invalid_state",
                            detail=detail,
                        )
                        return

                    from ...workflow_engine.bridge import RuntimeBridge
                    from ...workflow_engine.engine import _node_version_required_text

                    if not RuntimeBridge.check_node_available():
                        self._reply_workflow_error(
                            message_id,
                            "invalid_argument",
                            detail=_node_version_required_text(),
                        )
                        return

                    req_stripped = (req or args).strip()
                    if len(req_stripped) < 4:
                        self._reply_workflow_error(
                            message_id,
                            "invalid_argument",
                            detail="需求描述过短，请提供更详细的说明（至少 4 个字符）。",
                        )
                        return
                    if len(req_stripped) > 4000:
                        self._reply_workflow_error(
                            message_id,
                            "invalid_argument",
                            detail="需求描述过长（超过 4000 字符），请精简后重试。",
                        )
                        return

                    self._ensure_topic_engine_context(
                        mode="workflow",
                        message_id=message_id,
                        chat_id=chat_id,
                        project=project,
                    )
                    self.show_workflow_config(
                        message_id,
                        chat_id,
                        project,
                        requirement=req_stripped,
                    )
            else:
                self.show_workflow_config(message_id, chat_id, project)
        else:
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail=(
                    "未知命令。可用命令列表:\n"
                    "• `/wf <需求>` — 打开团队配置卡并确认启动 Workflow\n"
                    "• `/wf --auto <需求>` — 跳过配置卡，按默认团队极速启动\n"
                    "• `/wf_config` — 打开团队与 `tool + model + effort` 配置\n"
                    "• `/wf_status` — 查看当前 Workflow 进度\n"
                    "• `/wf_help` — 查看完整帮助\n"
                    "• `/stop_wf` — 停止正在执行的 Workflow\n"
                    "\n发送 `/wf_help` 查看完整说明。"
                ),
            )

    def _get_engine_manager(self):
        return self.ctx.workflow_engine_manager

    def _get_engine_name_prefix(self) -> str:
        return "Workflow"


    def _get_task_type(self) -> str:
        return "workflow_engine"

    def _show_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        self.show_workflow_status(message_id, chat_id, project)

    def _create_callbacks(
        self, message_id: str, chat_id: str, project: Optional["ProjectContext"], engine_name: str, root_path: str
    ):
        return self._build_workflow_callbacks(message_id, chat_id, project)

    # ------------------------------------------------------------------
    # Start workflow
    # ------------------------------------------------------------------

    def start_workflow(
        self,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: Optional["ProjectContext"] = None,
        *,
        preset_id: str | None = None,
        override_pool: tuple[Any, ...] | None = None,
        override_orchestrator_id: str | None = None,
        reuse_card_message_id: str | None = None,
    ) -> None:
        """Start a workflow and use the default auto policy to launch generation.

        The recommended available tool and its backend-default model are used;
        generation, validation, and execution proceed without a decision gate.
        """
        project = self._ensure_project(message_id, chat_id, project)
        if not project:
            return

        root_path = project.root_path if project else self.get_working_dir(chat_id)

        # Check for existing running workflow
        existing = self.ctx.workflow_engine_manager.get(chat_id, root_path)
        if existing and existing.is_running is True:
            self._reply_workflow_error(
                message_id,
                "invalid_state",
                detail="当前项目已有 Workflow 任务在执行中。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。",
            )
            return

        # Check Node.js availability
        from ...workflow_engine.bridge import RuntimeBridge
        from ...workflow_engine.engine import _node_version_required_text

        if not RuntimeBridge.check_node_available():
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail=_node_version_required_text(),
            )
            return

        # Input validation: requirement must be non-trivial
        _req_stripped = requirement.strip()
        if len(_req_stripped) < 4:
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail="需求描述过短，请提供更详细的说明（至少 4 个字符）。",
            )
            return
        if len(_req_stripped) > 4000:
            self._reply_workflow_error(
                message_id,
                "invalid_argument",
                detail="需求描述过长（超过 4000 字符），请精简后重试。",
            )
            return

        # Newest-wins applies only before the runtime thread has claimed the
        # reusable engine. The same initiator (or an admin) may discard an old
        # admission/generation/queued start; another user's task is protected.
        from ...thread import get_current_sender_id

        admission_engine = existing or self.ctx.workflow_engine_manager.get_or_create(
            chat_id,
            root_path,
            engine_name=self.get_engine_name(
                chat_id,
                project_id=(project.project_id if project else None),
            ),
        )
        (
            takeover_ok,
            takeover_error,
            admission_owner,
        ) = self._supersede_incomplete_workflow(
            admission_engine,
            root_path=root_path,
            current_user=get_current_sender_id() or "",
        )
        if not takeover_ok:
            if takeover_error == "forbidden":
                self._reply_workflow_error(
                    message_id,
                    "forbidden",
                    detail="只能替换自己发起的未完成 Workflow；管理员可代为处理。",
                )
            else:
                self._reply_workflow_error(
                    message_id,
                    "invalid_state",
                    detail=("当前项目已有 Workflow 任务在执行中。发送 `/wf_status` 查看进度，或 `/stop_wf` 停止任务。"),
                )
            return

        self.add_reaction(message_id, EmojiReaction.on_multi_task_start())

        # Bind engine mode to topic for auto-routing
        topic_anchor_id = message_id
        if reuse_card_message_id:
            try:
                topic_anchor_id = self._resolve_origin(message_id) or message_id
            except Exception:
                topic_anchor_id = message_id
        self._ensure_topic_engine_context(
            mode="workflow",
            message_id=topic_anchor_id,
            chat_id=chat_id,
            project=project,
        )

        self._launch_auto_bound_workflow(
            message_id=message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            engine=admission_engine,
            admission_owner=admission_owner,
            preset_id=preset_id,
            override_pool=override_pool,
            override_orchestrator_id=override_orchestrator_id,
            reuse_card_message_id=reuse_card_message_id,
        )

    def handle_workflow_agent_action(
        self,
        message_id: str,
        chat_id: str,
        project_id: str | None,
        value: dict[str, Any],
    ) -> None:
        """Reject legacy Agent Pool selection card callbacks cleanly."""
        del chat_id, project_id, value
        self._reply_workflow_error(
            message_id,
            "session_expired",
            detail="旧版 Agent 选择卡已停用，请直接发送 `/wf <需求>` 自动启动。",
        )

    def _launch_auto_bound_workflow(
        self,
        *,
        message_id: str,
        chat_id: str,
        requirement: str,
        project: "ProjectContext" | None,
        root_path: str,
        engine: Any,
        admission_owner: _WorkflowLifecycleOwner | None,
        preset_id: str | None = None,
        override_pool: tuple[Any, ...] | None = None,
        override_orchestrator_id: str | None = None,
        reuse_card_message_id: str | None = None,
    ) -> None:
        """Auto-bind available tools or saved chat/project Teamwork pool and launch."""
        from ...project import get_project_workflow_team_config
        from ...spec_engine.models import ReviewAgentBinding
        from ...thread import get_current_sender_id
        from ...workflow_engine.agent_pool import (
            WorkflowAgentBinding,
            freeze_agent_pool,
            select_auto_orchestrator,
        )
        from ...workflow_engine.models import (
            PendingWorkflow,
            WorkflowProject,
            WorkflowStatus,
        )
        from ...workflow_engine.tool_registry import (
            get_available_tools,
            validate_agent_binding_spec,
        )

        available_tools = get_available_tools(require_available=True)
        resolved_preset = preset_id
        agent_pool: tuple[WorkflowAgentBinding, ...] = ()
        orchestrator_binding: WorkflowAgentBinding | None = None
        orchestrator_was_auto = True

        if override_pool:
            valid_override: list[WorkflowAgentBinding] = []
            for binding in override_pool:
                ok_b, err_b = validate_agent_binding_spec(
                    binding.tool_name,
                    model_name=binding.model_name,
                    profile=binding.profile,
                    effort=binding.effort,
                    available_tools=available_tools,
                )
                if not ok_b:
                    if admission_owner is not None:
                        admission_owner.done_event.set()
                        release_owner = getattr(engine, "release_lifecycle_owner", None)
                        if callable(release_owner):
                            release_owner(admission_owner)
                    self._reply_workflow_error(
                        message_id,
                        "invalid_argument",
                        detail=f"{binding.agent_id}: {err_b or '工具或参数不可用'}",
                    )
                    return
                valid_override.append(binding)
            try:
                agent_pool = freeze_agent_pool(valid_override, require_nonempty=True)
            except ValueError as exc:
                if admission_owner is not None:
                    admission_owner.done_event.set()
                    release_owner = getattr(engine, "release_lifecycle_owner", None)
                    if callable(release_owner):
                        release_owner(admission_owner)
                self._reply_workflow_error(
                    message_id,
                    "invalid_argument",
                    detail=str(exc),
                )
                return
            if override_orchestrator_id:
                orchestrator_binding = next(
                    (b for b in agent_pool if b.agent_id == override_orchestrator_id),
                    None,
                )
            if orchestrator_binding is None:
                orchestrator_binding = select_auto_orchestrator(agent_pool)
            else:
                orchestrator_was_auto = False
        else:
            saved_cfg = get_project_workflow_team_config(project, chat_id)
            if saved_cfg:
                raw_bindings = saved_cfg.get("bindings") or []
                valid_saved: list[WorkflowAgentBinding] = []
                for item in raw_bindings:
                    b = WorkflowAgentBinding.from_dict(item)
                    if b is None:
                        continue
                    ok_b, _ = validate_agent_binding_spec(
                        b.tool_name,
                        model_name=b.model_name,
                        profile=b.profile,
                        effort=b.effort,
                        available_tools=available_tools,
                    )
                    if ok_b:
                        valid_saved.append(b)
                if valid_saved:
                    agent_pool = freeze_agent_pool(valid_saved, require_nonempty=True)
                    saved_orch_id = str(saved_cfg.get("orchestrator_agent_id") or "").strip()
                    orchestrator_binding = next(
                        (b for b in agent_pool if b.agent_id == saved_orch_id),
                        None,
                    )
                    if orchestrator_binding is None:
                        orchestrator_binding = select_auto_orchestrator(agent_pool)
                    else:
                        orchestrator_was_auto = False
                    if resolved_preset is None:
                        resolved_preset = str(saved_cfg.get("preset_id") or "").strip() or None

        if not agent_pool:
            from ...mode.backends import ALL_BACKEND_SPECS

            hidden_tools = frozenset(
                spec.mode_key
                for spec in ALL_BACKEND_SPECS
                if not spec.visible_in_tools_list
            )
            preferred_order = (
                "traex",
                "claude",
                "codex",
                "grok",
                "dsh",
                "agy",
                "aiden",
                "gemini",
                "coco",
            )
            selected_names = [t for t in preferred_order if t in available_tools][:3]
            if not selected_names:
                selected_names = [
                    t for t in available_tools if t not in hidden_tools
                ][:3]
            if not selected_names:
                if admission_owner is not None:
                    admission_owner.done_event.set()
                    release_owner = getattr(engine, "release_lifecycle_owner", None)
                    if callable(release_owner):
                        release_owner(admission_owner)
                self._reply_workflow_error(
                    message_id,
                    "invalid_state",
                    detail="当前环境未检测到可执行的 Workflow 工具。",
                )
                return

            agent_pool = freeze_agent_pool(
                WorkflowAgentBinding(
                    agent_id=f"A{idx}",
                    tool_name=name,
                    model_name=None,
                    display_name=str(available_tools[name] or name),
                )
                for idx, name in enumerate(selected_names, start=1)
            )
            orchestrator_binding = select_auto_orchestrator(agent_pool)
            orchestrator_was_auto = True

        assert orchestrator_binding is not None
        initiator_user_id = get_current_sender_id() or ""
        workflow_project_id = uuid.uuid4().hex
        selected_tools = [binding.tool_name for binding in agent_pool]
        effective_orch_model = (
            orchestrator_binding.effective_model_name()
            or orchestrator_binding.model_name
        )

        with engine._lock:
            wf_project = engine.project
            if not wf_project:
                wf_project = WorkflowProject()
                engine._project = wf_project
                if getattr(engine, "project", None) is None:
                    engine.project = wf_project
            if admission_owner is not None:
                admission_owner.stop_event.set()
                admission_owner.heartbeat_stop_event.set()
                admission_owner.done_event.set()
                self._retire_workflow_owner(engine, admission_owner)
            session_key = admission_owner.session_key if admission_owner is not None else uuid.uuid4().hex
            generation_owner = _WorkflowLifecycleOwner(
                session_key=session_key,
                initiator_user_id=initiator_user_id,
                chat_id=chat_id,
                project_id=workflow_project_id,
                root_path=root_path,
                source_script_path=self._new_workflow_script_path(root_path),
            )
            engine._script_generation_owner = generation_owner
            wf_project.pending = PendingWorkflow(
                requirement=requirement,
                initiator_user_id=initiator_user_id,
                engine_session_key=session_key,
                selected_tools=selected_tools,
                orchestrator_agent=orchestrator_binding.tool_name,
                orchestrator_binding=ReviewAgentBinding(
                    provider="workflow",
                    tool_name=orchestrator_binding.tool_name,
                    display_name=orchestrator_binding.display_name,
                    agent_type=orchestrator_binding.tool_name,
                    model_name=effective_orch_model,
                    model_display_name=orchestrator_binding.model_name,
                    use_default_model=(
                        orchestrator_binding.model_name is None
                        and orchestrator_binding.effort is None
                        and orchestrator_binding.profile is None
                    ),
                ),
                review_agents=[],
                auto_reviewer=True,
                agent_pool=agent_pool,
                orchestrator_agent_id=orchestrator_binding.agent_id,
                orchestrator_was_auto=orchestrator_was_auto,
                project_id=workflow_project_id,
                preset_id=resolved_preset,
            )
            clear_renderer = getattr(engine, "clear_stale_renderer_for_generation", None)
            if callable(clear_renderer):
                clear_renderer()
            engine._renderer_wf = None
            wf_project.status = WorkflowStatus.GENERATING_SCRIPT

        if admission_owner is not None:
            release_owner = getattr(engine, "release_lifecycle_owner", None)
            if callable(release_owner):
                release_owner(admission_owner)

        try:
            origin_message_id = self._resolve_origin(message_id)
        except Exception:
            origin_message_id = message_id

        generation_card_id: str | None = None
        try:
            card_data = self._workflow_generation_card_data(
                requirement=requirement,
                agent_pool=agent_pool,
                orchestrator_agent_id=orchestrator_binding.agent_id,
                orchestrator_was_auto=orchestrator_was_auto,
                chat_id=chat_id,
                project_id=(project.project_id if project else workflow_project_id),
            )
            with generation_owner.delivery_lock:
                generation_card_id = self._replace_or_send_workflow_rendered_card(
                    card_message_id=reuse_card_message_id,
                    chat_id=chat_id,
                    card_data=card_data,
                    origin_message_id=origin_message_id,
                )
                if generation_card_id:
                    object.__setattr__(generation_owner, "card_message_id", generation_card_id)
        except Exception:
            logger.debug("Failed to deliver initial workflow generation card", exc_info=True)

        self._schedule_generate_and_start_workflow(
            message_id=generation_card_id or message_id,
            chat_id=chat_id,
            requirement=requirement,
            project=project,
            root_path=root_path,
            selected_tools=selected_tools,
            expected_session_key=session_key,
            engine=engine,
        )

    # ------------------------------------------------------------------
    # Error surface unification
    # ------------------------------------------------------------------

    def _build_error_card(
        self,
        category: str,
        *,
        detail: str = "",
    ) -> dict[str, Any]:
        """Build a standardized error card using the unified four-category surface.

        The supplied *detail* is always passed through the shared sanitizer
        (``sanitize_for_reply``) so that file paths, tracebacks, and internal
        module names never leak to the user. Only the sanitized user-facing
        message is rendered; raw details are logged but never shown.

        Args:
            category: One of "session_expired", "invalid_state", "invalid_argument", "forbidden", "internal_error"
            detail: Optional raw detail string — sanitized before rendering.
        """
        from ...card.builders.core import CoreBuilder
        from ...card.ui_text import UI_TEXT
        from ...workflow_engine.errors import (
            _strip_internal_details,
        )

        title_key = f"workflow_error_{category}_title"
        body_key = f"workflow_error_{category}_body"

        title = UI_TEXT.get(title_key, "操作失败")
        # Strip anything that looks like an internal traceback / file path
        # from the raw *detail* before rendering.  We intentionally do NOT
        # use ``sanitize_for_reply`` here - that helper already wraps the
        # message in a category-specific template, which would collide with
        # the UI_TEXT body template below and produce a duplicated prefix.
        safe_detail = _strip_internal_details(detail or "")

        raw_body = UI_TEXT.get(body_key, "")
        if raw_body and "{detail}" in raw_body:
            # Don't use format() — avoid unexpected kwargs when raw_body has
            # other placeholders. Do a simple literal replace instead.
            body = raw_body.replace("{detail}", safe_detail)
        elif raw_body:
            # Template does not define a {detail} placeholder; still surface
            # a sanitized user-facing hint when one is available so the user
            # sees "脚本被篡改/验证失败"这类可操作信息 rather than a
            # generic "服务内部错误".
            if safe_detail and category != "internal_error":
                body = raw_body.rstrip() + "\n\n🔎 细节：" + safe_detail
            else:
                body = raw_body
        else:
            body = safe_detail or "操作失败，请稍后重试。"

        # Header color by category
        header_template = "red"  # default for errors

        card = CoreBuilder._wrap_card(
            header_title=title,
            header_template=header_template,
            elements=[
                {"tag": "markdown", "content": body},
            ],
        )
        card.setdefault("config", {})["update_multi"] = True
        return card

    def _reply_workflow_error(
        self,
        message_id: str,
        category: str,
        *,
        detail: str = "",
    ) -> None:
        """Reply with a standardized error card."""
        card = self._build_error_card(category, detail=detail)
        self.reply_card(message_id, card)

    def _replace_or_send_workflow_card(
        self,
        *,
        card_message_id: str | None,
        chat_id: str,
        card: dict[str, Any],
        origin_message_id: str | None = None,
    ) -> None:
        """Replace an existing workflow card, falling back to a new card.

        Generation cards are chat-sent cards, so a failed patch would otherwise
        leave the user looking at a stale "生成脚本中" card.
        """
        if card_message_id and self.update_card(card_message_id, card):
            return
        self.send_card_to_chat(
            chat_id,
            card,
            origin_message_id=origin_message_id,
        )

    @staticmethod
    def _build_workflow_card_from_renderer_data(card_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize WorkflowProgressRenderer output to a full Feishu card.

        Workflow renderer helpers return ``{"header": ..., "elements": ...}``
        because they are pure renderers. Handler delivery APIs expect a full
        CardKit 2.0 card, so keep that conversion at the handler boundary.
        """
        raw_header = card_data.get("header") if isinstance(card_data, dict) else None
        header_source = raw_header if isinstance(raw_header, dict) else {}

        raw_title = header_source.get("title")
        if isinstance(raw_title, dict):
            title_content = str(raw_title.get("content") or "Workflow")
            title_tag = str(raw_title.get("tag") or "plain_text")
        else:
            title_content = str(raw_title or "Workflow")
            title_tag = "plain_text"

        header: dict[str, Any] = {
            "title": {"tag": title_tag, "content": title_content},
            "template": str(header_source.get("template") or "blue"),
        }
        raw_subtitle = header_source.get("subtitle")
        if isinstance(raw_subtitle, dict):
            subtitle_content = str(raw_subtitle.get("content") or "")
            if subtitle_content:
                header["subtitle"] = {
                    "tag": str(raw_subtitle.get("tag") or "plain_text"),
                    "content": subtitle_content,
                }

        body = card_data.get("body") if isinstance(card_data, dict) else None
        if isinstance(body, dict) and isinstance(body.get("elements"), list):
            elements = list(body["elements"])
        else:
            root_elements = card_data.get("elements") if isinstance(card_data, dict) else None
            elements = list(root_elements) if isinstance(root_elements, list) else []

        summary_content = " ".join(title_content.split()) or "Workflow 运行状态"
        if len(summary_content) < 8:
            summary_content = f"Workflow · {summary_content}"
        if len(summary_content) > 60:
            summary_content = f"{summary_content[:59]}…"
        card = {
            "schema": "2.0",
            "config": {
                "compact_width": False,
                "wide_screen_mode": True,
                "update_multi": True,
                "summary": {"content": summary_content},
            },
            "header": header,
            "body": {"elements": elements},
        }
        from ...card.delivery.page_mutator import guard_card_payload

        return guard_card_payload(
            json.loads(
                json.dumps(
                    card,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8", errors="replace").decode("utf-8")
            )
        )

    def _replace_or_send_workflow_rendered_card(
        self,
        *,
        card_message_id: str | None,
        chat_id: str,
        card_data: dict[str, Any],
        origin_message_id: str | None = None,
        fallback_to_new: bool = True,
    ) -> str | None:
        """Replace a Workflow renderer card, falling back to a new card.

        Returns the message id that should receive future progress updates.
        """
        card = self._build_workflow_card_from_renderer_data(card_data)
        if card_message_id and self.update_card(card_message_id, card):
            return card_message_id
        if not fallback_to_new:
            return None
        return self.send_card_to_chat(
            chat_id,
            card,
            origin_message_id=origin_message_id,
        )

    def _show_initial_workflow_progress_card(
        self,
        *,
        card_message_id: str,
        chat_id: str,
        wf_project: Any,
        origin_message_id: str | None = None,
    ) -> str:
        """Switch the generation card to a running progress card immediately."""
        try:
            from ...workflow_engine.renderer import WorkflowProgressRenderer

            card_data = WorkflowProgressRenderer(wf_project).render_progress_card()
            return (
                self._replace_or_send_workflow_rendered_card(
                    card_message_id=card_message_id,
                    chat_id=chat_id,
                    card_data=card_data,
                    origin_message_id=origin_message_id,
                )
                or card_message_id
            )
        except Exception:
            logger.debug("Failed to show initial workflow progress card", exc_info=True)
            return card_message_id


    @staticmethod
    def _workflow_generation_card_data(
        *,
        requirement: str,
        agent_pool: Any,
        orchestrator_agent_id: str,
        orchestrator_was_auto: bool,
        current_activity: str | None = None,
        elapsed_seconds: int = 0,
        terminal_status: str | None = None,
        chat_id: str = "",
        project_id: str = "",
    ) -> dict[str, Any]:
        from ...workflow_engine.renderer import WorkflowGenerationRenderer

        return WorkflowGenerationRenderer(
            requirement=requirement,
            agent_pool=agent_pool,
            orchestrator_agent_id=orchestrator_agent_id,
            orchestrator_was_auto=orchestrator_was_auto,
            chat_id=chat_id,
            project_id=project_id,
        ).render(
            current_activity=current_activity,
            elapsed_seconds=elapsed_seconds,
            terminal_status=terminal_status,
        )


    def _resolve_project_from_id(self, project_id: str, chat_id: str) -> Optional["ProjectContext"]:
        """Resolve a ProjectContext from project_id, scoped to the chat.

        Uses ``project_manager.get_project_for_chat`` so that a forged
        ``project_id`` carried in a card action payload cannot be used
        to interact with projects the chat is not allowed to see. Returns
        ``None`` for unknown or invisible projects.
        """
        if not project_id:
            return None
        try:
            return self.ctx.project_manager.get_project_for_chat(project_id, chat_id)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @staticmethod
    def _live_workflow_card_message_id(engine: Any) -> str | None:
        from ...workflow_engine.models import WorkflowStatus

        def _pick_card_id() -> str | None:
            wf_project = getattr(engine, "project", None) or getattr(
                engine, "workflow_project", None
            )
            if getattr(wf_project, "status", None) == WorkflowStatus.GENERATING_SCRIPT:
                owner = vars(engine).get("_script_generation_owner")
                card_id = getattr(owner, "card_message_id", None)
                if isinstance(card_id, str) and card_id:
                    return card_id
                return None
            for attr in ("_workflow_start_owner", "_script_generation_owner"):
                owner = vars(engine).get(attr)
                card_id = getattr(owner, "card_message_id", None)
                if isinstance(card_id, str) and card_id:
                    return card_id
            return None

        lock = getattr(engine, "_lock", None)
        if lock is not None:
            with lock:
                return _pick_card_id()
        return _pick_card_id()

    def show_workflow_status(
        self,
        message_id: str,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show current workflow progress, patching the live card without spamming duplicates."""
        from ...workflow_engine.models import WorkflowStatus
        from .workflow_card_pages import _without_page_key

        root_path = self._get_root_path(chat_id, project)
        engine = self.ctx.workflow_engine_manager.get(chat_id, root_path)

        if not engine:
            self.reply_text(message_id, "当前没有 Workflow 任务。")
            return

        wf_project = None
        lock = getattr(engine, "_lock", None)
        if lock is not None:
            with lock:
                wf_project = getattr(engine, "project", None) or getattr(
                    engine, "workflow_project", None
                )
        else:
            wf_project = getattr(engine, "project", None) or getattr(
                engine, "workflow_project", None
            )

        live_card_id = self._live_workflow_card_message_id(engine)
        project_id = getattr(project, "project_id", "") or ""

        if getattr(wf_project, "status", None) == WorkflowStatus.GENERATING_SCRIPT:
            pending = getattr(wf_project, "pending", None)
            req = (
                (getattr(pending, "requirement", None) if pending is not None else None)
                or getattr(wf_project, "requirement", "")
                or ""
            )
            pool = tuple(getattr(pending, "agent_pool", ()) or ()) if pending is not None else ()
            orch_id = (getattr(pending, "orchestrator_agent_id", None) if pending is not None else None) or ""
            orch_auto = bool(getattr(pending, "orchestrator_was_auto", False)) if pending is not None else False
            created_at = getattr(pending, "created_at", None) if pending is not None else None
            elapsed_seconds = (
                max(0, int(time.time() - created_at))
                if isinstance(created_at, (int, float))
                else 0
            )
            gen_status_text = (
                f"⏳ Workflow 正在生成编排脚本…（已耗时 {elapsed_seconds}s）"
                if elapsed_seconds > 0
                else "⏳ Workflow 正在生成编排脚本…"
            )
            if live_card_id:
                gen_card_data = self._workflow_generation_card_data(
                    requirement=req,
                    agent_pool=pool,
                    orchestrator_agent_id=orch_id,
                    orchestrator_was_auto=orch_auto,
                    current_activity="正在生成并验证 Workflow 编排脚本…",
                    elapsed_seconds=elapsed_seconds,
                    chat_id=chat_id,
                    project_id=project_id,
                )
                origin_message_id = self._resolve_origin(message_id)
                delivered = self._replace_or_send_workflow_rendered_card(
                    card_message_id=live_card_id,
                    chat_id=chat_id,
                    card_data=gen_card_data,
                    origin_message_id=origin_message_id,
                    fallback_to_new=False,
                )
                if not delivered:
                    self.reply_text(message_id, gen_status_text)
                return
            self.reply_text(message_id, gen_status_text)
            return

        status_text = engine.get_status_text()
        card_pages = engine.get_progress_cards()
        is_running = self._workflow_is_running_for_card(chat_id, project)

        if card_pages:
            origin_message_id = self._resolve_origin(message_id)
            if live_card_id:
                primary_page = _without_page_key(card_pages[0])
                if is_running:
                    self._inject_workflow_stop_button(
                        primary_page,
                        chat_id,
                        project_id,
                        is_running=True,
                    )
                delivered = self._replace_or_send_workflow_rendered_card(
                    card_message_id=live_card_id,
                    chat_id=chat_id,
                    card_data=primary_page,
                    origin_message_id=origin_message_id,
                    fallback_to_new=False,
                )
                if not delivered:
                    self.reply_text(message_id, status_text)
                return

            if getattr(wf_project, "status", None) in {
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED,
            }:
                self.reply_text(message_id, status_text)
                return

            failed_pages: list[int] = []
            for page_index, card_data in enumerate(card_pages):
                delivered = self._replace_or_send_workflow_rendered_card(
                    card_message_id=None,
                    chat_id=chat_id,
                    card_data=_without_page_key(card_data),
                    origin_message_id=origin_message_id,
                )
                if not delivered:
                    failed_pages.append(page_index)
            if failed_pages:
                pages = ", ".join(str(index + 1) for index in failed_pages)
                self.reply_text(
                    message_id,
                    f"Workflow 结果页重投仍不完整（失败页：{pages}），可再次发送 `/wf_status` 重试。",
                )
            return

        self.reply_text(message_id, status_text)

    def show_workflow_help(
        self,
        message_id: str,
        chat_id: str = "",
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Show the Workflow usage guide as a structured Card JSON 2.0 card."""
        from ...card.builders.workflow import build_workflow_help_card
        from ...workflow_engine.engine import _node_version_required_text

        card = build_workflow_help_card(
            node_requirement_text=_node_version_required_text(),
            project_id=(project.project_id if project else None),
            chat_id=chat_id,
        )
        self.reply_card(message_id, card)






    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_root_path(self, chat_id: str, project: Optional["ProjectContext"]) -> str:
        """Resolve root_path from project or chat."""
        if project:
            return project.root_path
        return self.get_working_dir(chat_id)
