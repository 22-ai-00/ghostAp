"""Programming mode handlers — config-driven template for all programming modes.

The ``ProgrammingModeHandler`` captures the shared logic for direct programming
modes. Concrete subclasses declare configuration
attributes; the base class provides default implementations for all hooks.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

from ...acp.manager import ACPSessionManager
from ...card.ui_text import UI_TEXT
from ...mode import PROGRAMMING_MODE_VALUES, InteractionMode
from ...mode.backends import ALL_BACKEND_METAS, BACKEND_SPEC_BY_MODE
from ...project import ContextSourceMode
from ...utils.errors import get_error_detail, log_exception
from ..emoji import EmojiReaction
from ..message_formatter import FeishuMessageFormatter as fmt
from .base import BaseHandler
from .programming_execution import (
    _PROGRAMMING_COT_REQUEST_TIMEOUT_S as _PROGRAMMING_COT_REQUEST_TIMEOUT_S,
)
from .programming_execution import (
    _SAFE_FEISHU_APP_ID as _SAFE_FEISHU_APP_ID,
)
from .programming_execution import (
    ProgrammingExecutionMixin as ProgrammingExecutionMixin,
)
from .programming_support import (
    _BTW_EVENT_LIMIT as _BTW_EVENT_LIMIT,
)
from .programming_support import (
    _BTW_EVENT_TEXT_LIMIT as _BTW_EVENT_TEXT_LIMIT,
)
from .programming_support import (
    _BTW_SNAPSHOT_LIMIT as _BTW_SNAPSHOT_LIMIT,
)
from .programming_support import (
    _ActiveProgrammingRun as _ActiveProgrammingRun,
)
from .programming_support import (
    _append_execution_notice as _append_execution_notice,
)
from .programming_support import (
    _bounded_btw_text as _bounded_btw_text,
)
from .programming_support import (
    _configured_execution_windows as _configured_execution_windows,
)
from .programming_support import (
    _configured_finalization_reserve as _configured_finalization_reserve,
)
from .programming_support import (
    _configured_programming_idle_timeout as _configured_programming_idle_timeout,
)
from .programming_support import (
    _diagnostic_action as _diagnostic_action,
)
from .programming_support import (
    _execution_diagnostic_details as _execution_diagnostic_details,
)
from .programming_support import (
    _finalization_incomplete_reason as _finalization_incomplete_reason,
)
from .programming_support import (
    _format_btw_event as _format_btw_event,
)
from .programming_support import (
    _format_swarm_completion_notice as _format_swarm_completion_notice,
)
from .programming_support import (
    _has_reconciled_stale_child_history as _has_reconciled_stale_child_history,
)
from .programming_support import (
    _incomplete_notice as _incomplete_notice,
)
from .programming_support import (
    _log_final_unresolved_tool_trace as _log_final_unresolved_tool_trace,
)
from .programming_support import (
    _log_prompt_execution as _log_prompt_execution,
)
from .programming_support import (
    _was_already_streamed_on_main as _was_already_streamed_on_main,
)
from .programming_support import (
    build_programming_session_callbacks as build_programming_session_callbacks,
)

if TYPE_CHECKING:
    from ...agent_session import SyncSession
    from ...project import ProjectContext


def __getattr__(name: str) -> Any:
    if name == "AuthoritativeChildLifecycleProof":
        from ...acp.collaboration import AuthoritativeChildLifecycleProof

        globals()["AuthoritativeChildLifecycleProof"] = AuthoritativeChildLifecycleProof
        return AuthoritativeChildLifecycleProof
    if name in ("run_prompt_across_execution_windows", "run_prompt_with_continuation"):
        from ...acp import (
            run_prompt_across_execution_windows,
            run_prompt_with_continuation,
        )

        globals()["run_prompt_across_execution_windows"] = run_prompt_across_execution_windows
        globals()["run_prompt_with_continuation"] = run_prompt_with_continuation
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = logging.getLogger(__name__)
_MODEL_OVERRIDE_UNSET = object()
_BTW_STARTUP_TIMEOUT_S = 30.0
_BTW_PROMPT_TIMEOUT_S = 90


class ProgrammingModeHandler(ProgrammingExecutionMixin, BaseHandler):
    """Config-driven template base for all programming modes."""

    # ── Subclass MUST set these ──
    mode_name: str              # "Coco" / "Claude" / ...
    mode_emoji: str             # "🤖" / "🔮" / ...
    interaction_mode: InteractionMode
    mode_key: str               # "coco" / "claude" / ... — used for managers, project API
    context_source: ContextSourceMode

    # ── Optional overrides ──
    is_coco: bool = False
    thinking_text: str = ""

    _PROGRAMMING_MODE_KEYS = tuple(
        (spec.interaction_mode, spec.is_mode_predicate_name, spec.mode_key)
        for spec in ALL_BACKEND_METAS
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self._current_model: Optional[str] = None
        self._active_programming_runs: dict[
            tuple[str, str, str], _ActiveProgrammingRun
        ] = {}
        self._active_programming_runs_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    @staticmethod
    def _active_run_key(
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> tuple[str, str, str]:
        from ...thread import get_current_thread_id

        project_id = str(getattr(project, "project_id", "") or "")
        thread_id = str(get_current_thread_id() or "")
        return str(chat_id or ""), project_id, thread_id

    def _register_active_programming_run(
        self,
        chat_id: str,
        project: Optional["ProjectContext"],
        *,
        task_text: str,
        cwd: str,
    ) -> tuple[tuple[str, str, str], _ActiveProgrammingRun]:
        agent_type = self._get_agent_type_override(project) or self.mode_key
        state = _ActiveProgrammingRun(
            task_text=task_text,
            cwd=cwd,
            agent_type=agent_type,
            model_name=self._get_model_name_override(project),
        )
        key = self._active_run_key(chat_id, project)
        with self._active_programming_runs_lock:
            self._active_programming_runs[key] = state
        return key, state

    def _unregister_active_programming_run(
        self,
        key: tuple[str, str, str],
        state: _ActiveProgrammingRun,
    ) -> None:
        with self._active_programming_runs_lock:
            if self._active_programming_runs.get(key) is state:
                self._active_programming_runs.pop(key, None)

    def _get_active_programming_run(
        self,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
    ) -> _ActiveProgrammingRun | None:
        key = self._active_run_key(chat_id, project)
        with self._active_programming_runs_lock:
            return self._active_programming_runs.get(key)

    def handle_btw(
        self,
        message_id: str,
        chat_id: str,
        question: str,
        project: Optional["ProjectContext"] = None,
    ) -> None:
        """Answer a side question without contending for the main task lock."""
        active_run = self._get_active_programming_run(chat_id, project)
        if active_run is None:
            # /btw is a GhostAP control command, not a portable provider slash
            # command. An idle programming session can answer the question normally.
            self.handle_message(message_id, chat_id, question, project)
            return

        if not active_run.claim_btw():
            self.reply_text(message_id, UI_TEXT["system_btw_already_running"])
            return

        self.reply_text(message_id, UI_TEXT["system_btw_observing"])
        auxiliary_session = None
        try:
            from ...agent_session import (
                close_session_safely,
                create_auxiliary_session,
            )

            startup_timeout = min(
                _BTW_STARTUP_TIMEOUT_S,
                max(
                    1.0,
                    float(getattr(self.settings, "acp_startup_timeout", 30) or 30),
                ),
            )
            auxiliary_session = create_auxiliary_session(
                agent_type=active_run.agent_type,
                cwd=active_run.cwd,
                model_name=active_run.model_name,
                thread_id="btw-observer",
                startup_timeout=startup_timeout,
                startup_retries=1,
                startup_log_failures=False,
            )
            prompt = (
                "你是一个正在执行的编程任务的只读旁路观察员。"
                "请只根据下面的实时快照回答用户的临时问题，不参与或改变主任务。\n"
                "要求：不要调用工具；不要声称看到了快照之外的信息；证据不足时明确说明；"
                "用中文简洁回答，并区分事实与推断。\n\n"
                f"{active_run.snapshot()}\n\n"
                f"用户旁路问题: {_bounded_btw_text(question, 4000)}"
            )
            result = auxiliary_session.send_prompt(
                prompt,
                timeout=_BTW_PROMPT_TIMEOUT_S,
            )
            answer = str(getattr(result, "text", "") or "").strip()
            if not answer:
                answer = active_run.progress_fallback()
            self.reply_text(
                message_id,
                UI_TEXT["system_btw_answer"].format(answer=answer),
            )
        except Exception as exc:
            logger.warning(
                "[%s] /btw auxiliary observation failed: %s",
                self.mode_name,
                get_error_detail(exc),
            )
            self.reply_text(
                message_id,
                UI_TEXT["system_btw_fallback"].format(
                    progress=active_run.progress_fallback(),
                ),
            )
        finally:
            if auxiliary_session is not None:
                close_session_safely(auxiliary_session)
            active_run.release_btw()

    # ------------------------------------------------------------------
    # Config-driven default implementations (subclass may override)
    # ------------------------------------------------------------------
    def _get_session_manager(self) -> ACPSessionManager:
        ctx = self.ctx
        if hasattr(type(ctx), "get_manager") or "get_manager" in getattr(ctx, "__dict__", {}):
            mgr = ctx.get_manager(self.mode_key)
            if mgr is not None:
                return mgr
        managers = getattr(ctx, "__dict__", {}).get("managers") if hasattr(ctx, "__dict__") else None
        if isinstance(managers, Mapping):
            mgr = managers.get(self.mode_key)
            if mgr is not None:
                return mgr
        return getattr(ctx, f"{self.mode_key}_manager")

    def _replace_timed_out_session(
        self,
        *,
        chat_id: str,
        project,
        cwd: str,
        thread_id: str | None,
        timed_out_session: SyncSession,
        startup_budget_s: float,
    ) -> SyncSession:
        """Replace an ACP session whose timed-out prompt could not be drained."""
        project_id = project.project_id if project else None
        manager = self._get_session_manager()
        replacement_deadline = time.monotonic() + max(0.0, startup_budget_s)
        self._clear_snapshot_for_session(
            project,
            timed_out_session,
            persistence_timeout_s=min(1.0, max(0.0, startup_budget_s / 4)),
        )
        remaining_budget = replacement_deadline - time.monotonic()
        if remaining_budget <= 0:
            raise TimeoutError("ACP 安全收尾会话重建预算已耗尽")
        outcome = manager.replace_session(
            chat_id,
            cwd=cwd,
            expected_session=timed_out_session,
            startup_timeout=remaining_budget,
            project_id=project_id,
            agent_type_override=self._get_agent_type_override(project),
            model_name=self._get_model_name_override(project),
            thread_id=thread_id,
        )
        if not outcome.created:
            if outcome.session is None:
                raise RuntimeError(
                    "原会话已被移除；旧任务停止收尾，避免复活已结束任务"
                )
            raise RuntimeError(
                "并发新会话已接管；旧任务停止收尾，避免干扰新任务"
            )
        replacement = outcome.session
        if thread_id and project:
            self._register_thread_context(thread_id, chat_id, project, replacement)
        return replacement

    def _retire_finalization_session(
        self,
        *,
        chat_id: str,
        project,
        thread_id: str | None,
        active_session: SyncSession,
        retirement_budget_s: float = 30.0,
    ) -> None:
        """Prevent a timed-out session and any late child work from being reused."""
        project_id = project.project_id if project else None
        retirement_deadline = (
            time.monotonic() + max(0.0, retirement_budget_s)
        )
        self._clear_snapshot_for_session(
            project,
            active_session,
            persistence_timeout_s=min(
                1.0,
                max(0.0, retirement_budget_s / 4),
            ),
        )
        remaining_budget = retirement_deadline - time.monotonic()
        if remaining_budget <= 0:
            raise TimeoutError("ACP 安全收尾会话退休预算已耗尽")
        self._get_session_manager().retire_session(
            chat_id,
            project_id=project_id,
            thread_id=thread_id,
            expected_session=active_session,
            timeout=remaining_budget,
        )

    def _clear_snapshot_for_session(
        self,
        project,
        session: SyncSession,
        persistence_timeout_s: float | None = None,
    ) -> None:
        """Clear only a snapshot that still points at the retiring session."""
        if project is None:
            return
        session_id = str(getattr(session, "session_id", "") or "")
        cleared = bool(
            session_id
            and project.clear_programming_snapshot_if_matches(
                self.mode_key,
                session_id,
            )
        )
        if cleared:
            self._persist_project_context(
                project,
                timeout_s=persistence_timeout_s,
            )

    def _persist_project_context(
        self,
        project: "ProjectContext",
        *,
        timeout_s: float | None = None,
    ) -> bool:
        """Best-effort persistence for session snapshot tombstones."""
        result = [False]
        completed = threading.Event()

        def _persist() -> None:
            try:
                persist = getattr(
                    self.project_manager,
                    "persist_project_context",
                    None,
                )
                if callable(persist):
                    result[0] = bool(persist(project))
                    if not result[0]:
                        logger.error(
                            "programming session snapshot tombstone was not persisted"
                        )
            except Exception:
                logger.warning(
                    "failed to persist programming session snapshot",
                    exc_info=True,
                )
            finally:
                completed.set()

        if timeout_s is None:
            _persist()
            return result[0]
        worker = threading.Thread(
            target=_persist,
            daemon=True,
            name=f"persist-session-tombstone-{project.project_id}",
        )
        worker.start()
        if not completed.wait(timeout=max(0.0, timeout_s)):
            logger.warning(
                "programming session snapshot persistence exceeded %.3fs; "
                "continuing retirement while persistence finishes",
                timeout_s,
            )
            return False
        return result[0]

    def _is_in_this_mode(self, chat_id: str, project_id: Optional[str] = None) -> bool:
        return self.mode_manager.get_mode(chat_id, project_id) == self.interaction_mode

    def _is_in_opposite_mode(self, chat_id: str, project_id: Optional[str] = None) -> bool:
        return self._is_any_other_programming_mode(chat_id, project_id=project_id)

    def _exit_opposite_mode(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None, silent: bool = False):
        self._exit_other_programming_modes(message_id, chat_id, project=project, silent=silent)

    def _enter_mode_on_manager(self, chat_id: str, project_id: Optional[str] = None):
        self.mode_manager.enter_programming_mode(chat_id, self.interaction_mode, project_id=project_id)

    def _get_interaction_mode(self):
        return self.interaction_mode

    def _set_mode_on_project(self, project: "ProjectContext", active: bool, session_id: str = "", count: int = 0):
        if active:
            project.set_programming_mode(self.mode_key, True, session_id, count)
            if self.mode_key in PROGRAMMING_MODE_VALUES:
                project.acp_tool_name = self.mode_key
        else:
            project.set_programming_mode(self.mode_key, False)

    def _update_snapshot_on_project(self, project: "ProjectContext", query: str, count: int, session_id: str = ""):
        project.update_programming_snapshot(self.mode_key, query, count, session_id)

    def _get_model_name_override(self, project: Optional["ProjectContext"] = None) -> Optional[str]:
        if project and getattr(project, "acp_tool_name", "") == self.mode_key:
            return getattr(project, "acp_model_name", None)
        return getattr(self, "_current_model", None)

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @current_model.setter
    def current_model(self, value: Optional[str]):
        self._current_model = value

    # ------------------------------------------------------------------
    # dynamic agent overrides
    # ------------------------------------------------------------------
    def _get_agent_type_override(self, project: Optional["ProjectContext"] = None) -> Optional[str]:
        return None

    def _uses_claude_cli(self) -> bool:
        return False

    def _deactivate_other_project_modes(self, project: Optional["ProjectContext"]) -> None:
        if not project:
            return
        current = self._get_interaction_mode()
        for mode, _predicate, key in self._PROGRAMMING_MODE_KEYS:
            if mode != current:
                project.set_programming_mode(key, False)

    def _iter_other_programming_mode_entries(self):
        current = self._get_interaction_mode()
        for mode, predicate_name, handler_key in self._PROGRAMMING_MODE_KEYS:
            if mode != current:
                yield mode, predicate_name, handler_key

    def _is_any_other_programming_mode(self, chat_id: str, project_id: Optional[str] = None) -> bool:
        for _mode, predicate_name, _handler_key in self._iter_other_programming_mode_entries():
            predicate = getattr(self.mode_manager, predicate_name, None)
            if callable(predicate) and predicate(chat_id, project_id=project_id):
                return True
        return False

    def _exit_other_programming_modes(self, message_id: str, chat_id: str, project: Optional["ProjectContext"], silent: bool = False):
        _pid = project.project_id if project else None
        for mode, predicate_name, handler_key in self._iter_other_programming_mode_entries():
            predicate = getattr(self.mode_manager, predicate_name, None)
            if not callable(predicate) or not predicate(chat_id, project_id=_pid):
                continue
            handler = self.get_handler(handler_key)
            if handler and handler is not self and hasattr(handler, "exit_mode"):
                handler.exit_mode(message_id, chat_id, project=project, silent=silent)

    # ------------------------------------------------------------------
    # enter_mode
    # ------------------------------------------------------------------
    def enter_mode(
        self, message_id: str, chat_id: str, silent: bool = False, project: Optional["ProjectContext"] = None,
        thread_id: Optional[str] = None,
        *,
        model_override: object = _MODEL_OVERRIDE_UNSET,
        commit_project_state: bool = True,
        activate_mode: bool = True,
        exit_opposite_mode: bool = True,
        inherit_thread_context: bool = True,
    ) -> bool:
        from ...thread import get_current_thread_id

        project_id = project.project_id if project else None

        if thread_id is None and inherit_thread_context:
            thread_id = get_current_thread_id()
        if not thread_id and self._is_in_this_mode(chat_id, project_id=project_id):
            manager = self._get_session_manager()
            live_session = manager.get_session(
                chat_id,
                project_id=project_id,
            )
            if live_session is not None:
                if not silent:
                    info = manager.get_session_info(
                        chat_id,
                        project_id=project_id,
                    )
                    self.reply_text(
                        message_id,
                        fmt.format_warning(
                            UI_TEXT["mode_already_in_msg"].format(
                                name=self.mode_name,
                                info=info,
                            )
                        ),
                    )
                return True
            logger.warning(
                "[%s] mode is active but its session is missing; starting a fresh session: "
                "chat=%s project=%s",
                self.mode_name,
                chat_id[:12] if chat_id else "?",
                project_id or "-",
            )

        previous_mode = self.mode_manager.get_mode(chat_id, project_id=project_id)

        if (
            exit_opposite_mode
            and not thread_id
            and self._is_in_opposite_mode(chat_id, project_id=project_id)
        ):
            self._exit_opposite_mode(message_id, chat_id, project=project, silent=True)

        if not project:
            working_dir = self.get_working_dir(chat_id)
            try:
                project, is_new = self.project_manager.get_or_create_project_for_path(working_dir, chat_id)
                if is_new:
                    logger.info("自动创建项目: %s @ %s", project.project_name, project.root_path)
                project_id = project.project_id
            except Exception as e:
                log_exception(logger, "自动创建项目失败", e)

        working_dir = self.get_working_dir(chat_id)
        cwd = project.root_path if project else working_dir

        if project:
            valid, path_msg = self.project_manager.validate_project_path(project.project_id)
            if not valid:
                if not silent:
                    self.reply_text(message_id, UI_TEXT["mode_invalid_project_path"].format(msg=path_msg))
                return False

        startup_timeout = getattr(self.settings, "acp_startup_timeout", 20)
        agent_type_override = None
        model_name = None

        try:
            agent_type_override = self._get_agent_type_override(project)
            model_name = (
                self._get_model_name_override(project)
                if model_override is _MODEL_OVERRIDE_UNSET
                else model_override
            )
            session = self._get_session_manager().ensure_session(
                chat_id,
                cwd=cwd,
                startup_timeout=startup_timeout,
                project_id=project_id,
                agent_type_override=agent_type_override,
                model_name=model_name,
                thread_id=thread_id,
            )
        except TimeoutError as e:
            # silent 路径也必须记录日志，否则上层 handle_message recovery 看到 session=None
            # 时只能输出泛化的 mode_session_fail_msg，根因（启动超时/失败）完全被吞掉。
            logger.warning(
                "[%s] enter_mode session startup timed out (silent=%s): %s",
                self.mode_name, silent, get_error_detail(e),
            )
            if not silent:
                self.send_error_card(
                    chat_id,
                    e,
                    title=UI_TEXT["mode_startup_timeout_title"].format(name=self.mode_name),
                    origin_message_id=message_id,
                )
            return False
        except Exception as e:
            logger.warning(
                "[%s] enter_mode session startup failed (silent=%s): %s",
                self.mode_name, silent, get_error_detail(e),
                exc_info=True,
            )
            if not silent:
                self.send_error_card(
                    chat_id,
                    e,
                    title=UI_TEXT["mode_startup_fail_title"].format(name=self.mode_name),
                    origin_message_id=message_id,
                )
            return False

        if activate_mode and not thread_id:
            self._enter_mode_on_manager(chat_id, project_id=project_id)
        self.add_reaction(message_id, EmojiReaction.on_coco_enter())

        if project:
            if not thread_id and commit_project_state:
                self._deactivate_other_project_modes(project)
                self._set_mode_on_project(project, True, session.session_id)
            if not silent:
                from ...card.builders.core import CoreBuilder
                from ...card.builders.project import ProjectBuilder

                content = UI_TEXT["mode_enter_msg"].format(emoji=self.mode_emoji, name=self.mode_name)

                banner = CoreBuilder._build_banner_element(UI_TEXT["mode_enter_banner"].format(name=self.mode_name), type="success")
                msg_type, card_content = ProjectBuilder.build_project_response_card(
                    project,
                    UI_TEXT["mode_card_programming_title"].format(emoji=self.mode_emoji, name=self.mode_name),
                    content,
                    show_buttons=True,
                    footer=UI_TEXT["mode_project_dir_label"].format(path=project.root_path),
                    banner=banner,
                )
                response_id = self.reply_card(message_id, card_content)
                if response_id:
                    self.register_message_project(response_id, project)
        else:
            if not silent:
                if self.is_coco:
                    self.reply_text(message_id, fmt.format_coco_enter())
                else:
                    self.reply_text(
                        message_id,
                        UI_TEXT["mode_enter_no_project_msg"].format(emoji=self.mode_emoji, name=self.mode_name),
                    )

        if project and commit_project_state:
            self.record_mode_transition(
                project.project_id,
                previous_mode,
                self._get_interaction_mode(),
                reason=f"enter_{self.mode_name.lower()}_mode",
                chat_id=chat_id,
            )
        return True

    # ------------------------------------------------------------------
    # switch_model — live model switch for an active session
    # ------------------------------------------------------------------
    def switch_model(
        self,
        message_id: str,
        chat_id: str,
        model_name: Optional[str],
        project: Optional["ProjectContext"] = None,
        *,
        expected_session: SyncSession | None = None,
    ) -> bool:
        """Switch the model for the active programming session.

        Active sessions only use ACP ``session/setModel``. Session replacement
        belongs to task startup/recovery, not to this configuration command.
        The caller owns the selection card lifecycle, so this method is
        deliberately effect-only and never emits a second success/error card.
        """
        project_id = project.project_id if project else None
        mgr = self._get_session_manager()
        if self.mode_key == "traex":
            backend_model_name = model_name
        else:
            from ...acp.providers import normalize_acp_model_name

            backend_model_name = normalize_acp_model_name(self.mode_key, model_name)
        if backend_model_name != model_name:
            logger.info(
                "[%s] Normalized selected model for backend: selected=%s backend=%s",
                self.mode_name,
                model_name,
                backend_model_name,
            )

        session = mgr.get_session(chat_id, project_id=project_id)
        set_model_fn = getattr(session, "set_model", None) if session else None
        try:
            if expected_session is not None and session is not expected_session:
                raise RuntimeError("ACP 会话已变化，本次模型切换已失效")
            if not backend_model_name or not callable(set_model_fn):
                raise RuntimeError("当前 ACP 会话不支持在线切换模型")
            if not set_model_fn(backend_model_name):
                raise RuntimeError("ACP 后端拒绝了模型切换")
            logger.info(
                "[%s] Model switched via ACP protocol: %s",
                self.mode_name,
                backend_model_name,
            )
            return True
        except Exception as e:
            from ...utils.errors import log_exception
            log_exception(logger, f"切换 {self.mode_name} 模型失败", e)
            return False

    # ------------------------------------------------------------------
    # Thread context registration
    # ------------------------------------------------------------------
    def _register_thread_context(
        self,
        thread_root_id: str,
        chat_id: str,
        project: Optional["ProjectContext"],
        session: SyncSession,
        alias_keys: Optional[list[str]] = None,
    ) -> None:
        try:
            from ...thread import get_thread_manager

            mode_name = self._get_interaction_mode().value
            tool_name = None
            model_name = None
            if project:
                project_id = project.project_id
            else:
                active = self.project_manager.get_active_project(chat_id)
                project_id = active.project_id if active else (session.session_id or "unknown")
                if active:
                    project = active
            if project and not tool_name and getattr(project, "acp_tool_name", None):
                tool_name = project.acp_tool_name
            if project and not model_name and getattr(project, "acp_model_name", None):
                model_name = project.acp_model_name

            get_thread_manager().register(
                thread_root_id=thread_root_id,
                chat_id=chat_id,
                project_id=project_id,
                mode=mode_name,
                tool_name=tool_name,
                model_name=model_name,
                alias_keys=alias_keys,
            )
        except Exception as e:
            logger.warning("[Thread] Failed to register context: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # exit_mode
    # ------------------------------------------------------------------
    def exit_mode(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None, silent: bool = False):
        from ...thread import get_current_thread_id, get_thread_manager

        project_id = project.project_id if project else None
        thread_id = get_current_thread_id()
        session = self._get_session_manager().get_session(chat_id, project_id=project_id, thread_id=thread_id)

        # Capture before exit_to_smart resets mode state
        was_in_this_mode = self._is_in_this_mode(chat_id, project_id=project_id)

        is_pending_slot = (
            not thread_id
            and not session
            and self.settings.thread_programming_enabled
            and was_in_this_mode
        )
        # User is in this mode (mode_manager) but has no active session (e.g. entered mode without sending any message)
        is_mode_only_exit = (
            not thread_id
            and not session
            and not self.settings.thread_programming_enabled
            and was_in_this_mode
        )

        project_state_persisted = True
        if project:
            if session:
                self._update_snapshot_on_project(
                    project,
                    query=session.last_query,
                    count=session.message_count,
                    session_id=session.session_id,
                )
                self.context_manager.update_context(
                    project.project_id,
                    session_snapshot={
                        "data": session.to_snapshot(),
                        "source_mode": self.context_source.value,
                    },
                    chat_id=chat_id,
                )
            if not thread_id:
                self._set_mode_on_project(project, False)
                persist = getattr(
                    self.project_manager,
                    "persist_project_context",
                    None,
                )
                try:
                    project_state_persisted = not callable(persist) or bool(
                        persist(project)
                    )
                except Exception:
                    project_state_persisted = False
                    logger.exception(
                        "[%s] project mode exit persistence raised: project=%s",
                        self.mode_name,
                        project.project_id,
                    )
                if not project_state_persisted:
                    logger.error(
                        "[%s] failed to persist project mode exit: project=%s",
                        self.mode_name,
                        project.project_id,
                    )

        if not thread_id:
            self.mode_manager.exit_to_smart(chat_id, project_id=project_id)

        try:
            manager = self._get_session_manager()
            if session:
                # Ask the live backend to stop before retiring its transport.
                # ``end_session`` still owns close/removal and is deliberately
                # not replaced by cancellation.
                manager.cancel_session(
                    chat_id,
                    project_id=project_id,
                    thread_id=thread_id,
                    user_initiated=not silent,
                )
            has_session = manager.end_session(chat_id, project_id=project_id, thread_id=thread_id)
            if silent:
                # Silent mode: skip all user-facing messages (used for automatic mode switching)
                if has_session or is_pending_slot or is_mode_only_exit:
                    self.add_reaction(message_id, EmojiReaction.on_coco_exit())
                return
            if has_session or is_pending_slot or is_mode_only_exit:
                self.add_reaction(message_id, EmojiReaction.on_coco_exit())

                if project:
                    if not project_state_persisted:
                        content = UI_TEXT["mode_exit_persist_failed"]
                        banner_text = UI_TEXT[
                            "mode_exit_persist_failed_banner"
                        ].format(name=self.mode_name)
                        banner_type = "warning"
                        card_title = UI_TEXT[
                            "mode_exit_persist_failed_title"
                        ]
                    elif is_pending_slot or is_mode_only_exit:
                        content = UI_TEXT["mode_exit_pending_msg"].format(name=self.mode_name)
                        banner_text = UI_TEXT["mode_exit_banner"].format(
                            name=self.mode_name
                        )
                        banner_type = "info"
                        card_title = UI_TEXT["mode_exit_card_title"]
                    else:
                        content = UI_TEXT["mode_exit_msg"].format(
                            name=self.mode_name
                        )
                        banner_text = UI_TEXT["mode_exit_banner"].format(
                            name=self.mode_name
                        )
                        banner_type = "info"
                        card_title = UI_TEXT["mode_exit_card_title"]

                    from ...card.builders.core import CoreBuilder
                    from ...card.builders.project import ProjectBuilder

                    banner = CoreBuilder._build_banner_element(
                        banner_text,
                        type=banner_type,
                    )
                    msg_type, card_content = ProjectBuilder.build_project_response_card(
                        project,
                        card_title,
                        content,
                        show_buttons=True,
                        banner=banner,
                    )
                    response_id = self.reply_card(
                        message_id, card_content,
                        reply_in_thread=True if thread_id else None,
                    )
                    if response_id:
                        self.register_message_project(response_id, project)
                else:
                    self.reply_text(
                        message_id,
                        UI_TEXT["mode_exit_pending_msg"].format(name=self.mode_name),
                        reply_in_thread=True if thread_id else None,
                    )
            else:
                self.reply_text(
                    message_id,
                    fmt.format_warning(UI_TEXT["mode_not_in_msg"].format(name=self.mode_name)),
                    reply_in_thread=True if thread_id else None,
                )
        finally:
            if thread_id is not None:
                get_thread_manager().remove(thread_id)

    def stop_active_run(
        self,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        *,
        thread_id: Optional[str] = None,
    ) -> bool:
        """Cancel the running prompt immediately without exiting programming mode."""
        from ...thread import get_current_thread_id

        if thread_id is None:
            thread_id = get_current_thread_id()
        pid = project.project_id if project else None
        cancelled = False

        active_run = self._get_active_programming_run(chat_id, project)
        if active_run is not None:
            cancelled = True
            prog_session = getattr(active_run, "prog_session", None)
            if prog_session is not None and callable(getattr(prog_session, "mark_stopping", None)):
                try:
                    prog_session.mark_stopping()
                except Exception:
                    logger.debug("Failed to mark active programming card stopping", exc_info=True)
            run_session = getattr(active_run, "session", None)
            if run_session is not None and callable(getattr(run_session, "cancel", None)):
                active_gen = getattr(
                    run_session,
                    "active_prompt_generation",
                    lambda: None,
                )()
                mark_user_cancel = getattr(run_session, "mark_user_cancel", None)
                if active_gen is not None and callable(mark_user_cancel):
                    try:
                        mark_user_cancel(active_gen)
                    except Exception:
                        logger.debug("Failed to mark active programming run user cancel", exc_info=True)
                try:
                    run_session.cancel(wait=False)
                except Exception:
                    logger.debug("Failed to cancel active programming run session", exc_info=True)

        manager = self._get_session_manager()
        if manager is not None and callable(getattr(manager, "cancel_session", None)):
            try:
                try:
                    cancelled_mgr = manager.cancel_session(
                        chat_id,
                        project_id=pid,
                        thread_id=thread_id,
                        wait=False,
                        user_initiated=True,
                    )
                except TypeError:
                    cancelled_mgr = manager.cancel_session(
                        chat_id,
                        project_id=pid,
                        thread_id=thread_id,
                        wait=False,
                    )
                if cancelled_mgr:
                    cancelled = True
            except Exception:
                logger.debug("Failed to cancel managed programming session", exc_info=True)
        return cancelled

    def clear_session_context(
        self,
        chat_id: str,
        project: Optional["ProjectContext"] = None,
        *,
        thread_id: Optional[str] = None,
    ) -> None:
        """Reset conversation & live session while preserving mode, tool, and model."""
        from ...thread import get_current_thread_id

        if thread_id is None:
            thread_id = get_current_thread_id()
        pid = project.project_id if project else None
        manager = self._get_session_manager()
        if manager is not None and callable(getattr(manager, "end_session", None)):
            try:
                manager.end_session(chat_id, project_id=pid, thread_id=thread_id)
            except Exception:
                logger.debug("Failed to end session during clear_session_context", exc_info=True)
        if project is not None:
            if hasattr(project, "clear_session_state"):
                project.clear_session_state(self.mode_key)
            store = getattr(self.context_manager, "store", None)
            if store is not None and callable(getattr(store, "clear", None)):
                try:
                    store.clear(pid, chat_id=chat_id)
                except TypeError:
                    pass

    # ------------------------------------------------------------------
    # show_info
    # ------------------------------------------------------------------
    def show_info(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        from ...thread import get_current_thread_id

        project_id = project.project_id if project else None
        thread_id = get_current_thread_id()
        info = self._get_session_manager().get_session_info(chat_id, project_id=project_id, thread_id=thread_id)
        if info:
            if project:
                from ...card.builders.project import ProjectBuilder

                msg_type, card_content = ProjectBuilder.build_project_response_card(
                    project,
                    UI_TEXT["mode_session_info_title"].format(name=self.mode_name),
                    info,
                    show_buttons=True,
                )
                response_id = self.reply_card(message_id, card_content)
                if response_id:
                    self.register_message_project(response_id, project)
            else:
                self.reply_text(message_id, info)
        else:
            self.reply_text(message_id, fmt.format_warning(UI_TEXT["mode_not_in_msg"].format(name=self.mode_name)))

    # ------------------------------------------------------------------
    # Card actions
    # ------------------------------------------------------------------
    def handle_card_exit(self, message_id: str, chat_id: str, project_id: str, value: Optional[dict] = None):
        if project_id:
            project = self.project_manager.get_project_for_chat(project_id, chat_id)
        else:
            project = None
        system = self.get_handler("system")
        if system is not None and hasattr(system, "exit_current_mode"):
            system.exit_current_mode(message_id, chat_id, project=project)
            return
        self.exit_mode(message_id, chat_id, project=project)


# ======================================================================
# Concrete subclasses (SSOT spec-driven bindings & auto-registration)
# ======================================================================


PROGRAMMING_HANDLER_TYPES: dict[str, type[ProgrammingModeHandler]] = {}


def _bind_programming_backend(
    interaction_mode: InteractionMode,
    *,
    context_source: ContextSourceMode | None = None,
):
    """Attach canonical SSOT backend metadata and register the handler class."""
    spec = BACKEND_SPEC_BY_MODE[interaction_mode]
    resolved_source = context_source or ContextSourceMode(spec.mode_key)

    def _decorator(cls: type[ProgrammingModeHandler]) -> type[ProgrammingModeHandler]:
        cls.mode_name = spec.display_name
        cls.mode_emoji = spec.mode_emoji
        cls.interaction_mode = spec.interaction_mode
        cls.mode_key = spec.mode_key
        cls.context_source = resolved_source
        cls.thinking_text = UI_TEXT["mode_thinking_msg"].format(
            emoji=spec.mode_emoji,
            name=spec.display_name,
        )
        if spec.is_coco:
            cls.is_coco = True
        if spec.uses_claude_cli:
            cls._uses_claude_cli = lambda self: True
        PROGRAMMING_HANDLER_TYPES[spec.mode_key] = cls
        return cls

    return _decorator


@_bind_programming_backend(InteractionMode.COCO)
class CocoModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.CLAUDE)
class ClaudeModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.CLAUDE_W)
class ClaudeWModeHandler(ClaudeModeHandler):
    pass


@_bind_programming_backend(InteractionMode.AIDEN)
class AidenModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.CODEX)
class CodexModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.CODEX_W)
class CodexWModeHandler(CodexModeHandler):
    pass


@_bind_programming_backend(InteractionMode.GEMINI)
class GeminiModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.TRAEX)
class TraexModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.GROK)
class GrokModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.DSH)
class DSHModeHandler(ProgrammingModeHandler):
    pass


@_bind_programming_backend(InteractionMode.AGY)
class AGYModeHandler(ProgrammingModeHandler):
    pass
