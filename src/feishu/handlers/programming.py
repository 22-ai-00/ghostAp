"""Programming mode handlers — config-driven template for all programming modes.

The ``ProgrammingModeHandler`` captures the shared logic for direct programming
modes. Concrete subclasses declare configuration
attributes; the base class provides default implementations for all hooks.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from ...acp import (
    ACPEventRenderer,
    run_prompt_across_execution_windows,
    run_prompt_with_continuation,
)
from ...acp.collaboration import AuthoritativeChildLifecycleProof
from ...acp.manager import ACPSessionManager
from ...acp.outcome import PromptAssessment, PromptOutcome
from ...acp.providers import normalize_acp_model_name
from ...agent_session import SyncSession
from ...card.actions import dispatch as card_action_ids
from ...card.builders.core import CoreBuilder
from ...card.builders.project import ProjectBuilder
from ...card.builders.system import SystemBuilder
from ...card.hooks import EmojiHook
from ...card.session.config import SessionCallbacks
from ...card.ui_text import UI_TEXT
from ...mode import InteractionMode
from ...project import ContextSourceMode
from ...utils.errors import get_error_detail, log_exception
from ..emoji import EmojiReaction
from ..message_formatter import FeishuMessageFormatter as fmt
from .base import BaseHandler

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)
_MODEL_OVERRIDE_UNSET = object()


def _append_execution_notice(text: str, notice: str) -> str:
    """Append one terminal diagnostic without duplicating backend output."""
    content = str(text or "").strip()
    notice = str(notice or "").strip()
    if not notice or notice in content:
        return content
    return f"{content}\n\n{notice}" if content else notice


def _configured_finalization_reserve(settings: object) -> int:
    raw = getattr(settings, "programming_finalization_reserve_s", 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    return max(0, int(raw))


def _configured_execution_windows(settings: object) -> int:
    raw = getattr(settings, "programming_max_execution_windows", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 1
    return min(24, max(1, raw))


def _finalization_incomplete_reason(
    assessment: PromptAssessment,
) -> str:
    if assessment.pending_plan_entries > 0:
        return f"仍有 {assessment.pending_plan_entries} 个计划项未完成"
    if assessment.incomplete_tool_calls > 0:
        return (
            f"仍有 {assessment.incomplete_tool_calls} 个工具调用未进入终态"
        )
    return assessment.detail


def _incomplete_notice(execution: object) -> str:
    assessment = execution.assessment
    if getattr(execution, "window_limit_reached", False):
        return UI_TEXT["mode_exec_window_limit_incomplete_msg"].format(
            windows=max(1, int(getattr(execution, "execution_windows", 1))),
            reason=_finalization_incomplete_reason(assessment),
        )
    if not execution.entered_finalization:
        return UI_TEXT["mode_exec_incomplete_msg"].format(
            reason=assessment.detail,
        )
    goal = getattr(execution.result, "goal", None)
    key = (
        "mode_exec_finalization_paused_goal_msg"
        if goal is not None and goal.status == "paused"
        else "mode_exec_finalization_incomplete_msg"
    )
    return UI_TEXT[key].format(
        reason=_finalization_incomplete_reason(assessment),
    )


def _log_prompt_execution(mode_name: str, execution: object) -> None:
    assessment = execution.assessment
    goal = getattr(execution.result, "goal", None)
    raw_goal_status = getattr(goal, "status", None)
    goal_status = (
        raw_goal_status.strip().casefold()
        if isinstance(raw_goal_status, str)
        and raw_goal_status.strip().casefold()
        in {"active", "paused", "blocked", "completed"}
        else "unknown"
    )
    logger.info(
        "%s ACP执行判定: entered_finalization=%s execution_windows=%s "
        "window_limit_reached=%s goal_status=%s "
        "automatic_continuations=%s pending_plan_entries=%s "
        "incomplete_tool_calls=%s incomplete_outer_tool_calls=%s "
        "unresolved_child_tool_calls=%s unresolved_tools=%s",
        mode_name,
        execution.entered_finalization,
        getattr(execution, "execution_windows", 1),
        getattr(execution, "window_limit_reached", False),
        goal_status,
        execution.automatic_continuations,
        assessment.pending_plan_entries,
        assessment.incomplete_tool_calls,
        assessment.incomplete_outer_tool_calls,
        assessment.unresolved_child_tool_calls,
        ";".join(assessment.incomplete_tool_diagnostics) or "none",
    )


def _execution_diagnostic_details(execution: object) -> str:
    """Return a bounded, provider-allowlisted diagnostic for card callbacks."""
    assessment = execution.assessment
    diagnostics = ";".join(assessment.incomplete_tool_diagnostics) or "none"
    return "\n".join(
        (
            f"outcome={assessment.outcome.value}",
            f"stop_reason={assessment.stop_reason or 'unknown'}",
            f"execution_windows={max(1, int(getattr(execution, 'execution_windows', 1)))}",
            f"entered_finalization={bool(execution.entered_finalization)}",
            f"automatic_continuations={max(0, int(execution.automatic_continuations))}",
            f"pending_plan_entries={assessment.pending_plan_entries}",
            f"incomplete_tool_calls={assessment.incomplete_tool_calls}",
            f"incomplete_outer_tool_calls={assessment.incomplete_outer_tool_calls}",
            f"unresolved_child_tool_calls={assessment.unresolved_child_tool_calls}",
            f"unresolved_tools={diagnostics}",
        )
    )


def _diagnostic_action(chat_id: str, message_id: str) -> dict[str, str]:
    return {
        "action": card_action_ids.SHOW_ERROR_DETAILS,
        "chat_id": chat_id,
        "origin_message_id": message_id,
    }


def _has_reconciled_stale_child_history(
    execution: object,
    child_lifecycle_proof: AuthoritativeChildLifecycleProof,
) -> bool:
    """Accept only a final explicit list_agents proof, never a display inference."""
    if execution.entered_finalization:
        return False
    if getattr(execution, "window_limit_reached", False):
        return False
    assessment = execution.assessment
    if (
        assessment.outcome is not PromptOutcome.INCOMPLETE
        or assessment.stop_reason != "end_turn"
        or assessment.pending_plan_entries != 0
        or assessment.incomplete_outer_tool_calls != 0
        or assessment.unresolved_child_tool_calls <= 0
        or assessment.incomplete_tool_calls
        != assessment.unresolved_child_tool_calls
    ):
        return False
    goal = getattr(execution.result, "goal", None)
    raw_goal_status = getattr(goal, "status", "") if goal is not None else ""
    goal_status = str(getattr(raw_goal_status, "value", raw_goal_status) or "").strip().casefold()
    if goal is not None and goal_status != "completed":
        return False
    return child_lifecycle_proof.all_observed_children_terminal()


def build_programming_session_callbacks(
    *,
    reply_text_fn: Callable[[str, str], object],
    add_reaction: Callable[[str, str], object],
    message_id: str,
    chat_id: str,
) -> SessionCallbacks:
    """Build callbacks for normal programming CardSession lifecycle."""
    return SessionCallbacks(
        reply_text_fn=reply_text_fn,
        hooks=(
            EmojiHook(
                add_reaction=add_reaction,
                message_id=message_id,
                chat_id=chat_id,
            ),
        ),
    )


class ProgrammingModeHandler(BaseHandler):
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

    _PROGRAMMING_MODE_KEYS = (
        (InteractionMode.COCO, "is_coco_mode", "coco"),
        (InteractionMode.CLAUDE, "is_claude_mode", "claude"),
        (InteractionMode.AIDEN, "is_aiden_mode", "aiden"),
        (InteractionMode.CODEX, "is_codex_mode", "codex"),
        (InteractionMode.GEMINI, "is_gemini_mode", "gemini"),
        (InteractionMode.TRAEX, "is_traex_mode", "traex"),
        (InteractionMode.GROK, "is_grok_mode", "grok"),
    )

    def __init__(self, ctx):
        super().__init__(ctx)
        self._current_model: Optional[str] = None

    # ------------------------------------------------------------------
    # Config-driven default implementations (subclass may override)
    # ------------------------------------------------------------------
    def _get_session_manager(self) -> ACPSessionManager:
        return getattr(self.ctx, f"{self.mode_key}_manager")

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
            if self.mode_key in {"coco", "claude", "aiden", "codex", "gemini", "traex", "grok"}:
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
    ) -> bool:
        from ...thread import get_current_thread_id

        project_id = project.project_id if project else None

        if thread_id is None:
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
    ) -> bool:
        """Switch the model for the active programming session.

        Active sessions only use ACP ``session/setModel``. Session replacement
        belongs to task startup/recovery, not to this configuration command.
        """
        project_id = project.project_id if project else None
        mgr = self._get_session_manager()
        backend_model_name = (
            model_name
            if self.mode_key == "traex"
            else normalize_acp_model_name(self.mode_key, model_name)
        )
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
            if not backend_model_name or not callable(set_model_fn):
                raise RuntimeError("当前 ACP 会话不支持在线切换模型")
            if not set_model_fn(backend_model_name):
                raise RuntimeError("ACP 后端拒绝了模型切换")
            logger.info(
                "[%s] Model switched via ACP protocol: %s",
                self.mode_name,
                backend_model_name,
            )
            banner = CoreBuilder._build_banner_element(
                UI_TEXT["mode_model_switched_banner"].format(
                    name=self.mode_name,
                    model=model_name,
                ),
                type="success",
            )
            msg_type, card_content = ProjectBuilder.build_project_response_card(
                project,
                UI_TEXT["mode_model_switched_title"].format(name=self.mode_name),
                UI_TEXT["mode_model_switch_context_kept"],
                banner=banner,
            )
            self.reply_card(message_id, card_content)
            return True
        except Exception as e:
            from ...utils.errors import log_exception
            log_exception(logger, f"切换 {self.mode_name} 模型失败", e)
            self.reply_error(message_id, UI_TEXT["mode_model_switch_error"].format(name=self.mode_name, error=get_error_detail(e)))
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
                    content = UI_TEXT["mode_exit_msg"].format(name=self.mode_name)
                    if is_pending_slot or is_mode_only_exit:
                        content = UI_TEXT["mode_exit_pending_msg"].format(name=self.mode_name)

                    banner = CoreBuilder._build_banner_element(UI_TEXT["mode_exit_banner"].format(name=self.mode_name), type="info")
                    msg_type, card_content = ProjectBuilder.build_project_response_card(
                        project,
                        UI_TEXT["mode_exit_card_title"],
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

    # ------------------------------------------------------------------
    # handle_message
    # ------------------------------------------------------------------
    def handle_message(self, message_id: str, chat_id: str, text: str, project: Optional["ProjectContext"] = None):
        from ...thread import get_current_thread_id

        project_id = project.project_id if project else None
        thread_id = get_current_thread_id()
        session = self._get_session_manager().get_session(chat_id, project_id=project_id, thread_id=thread_id)

        if not session:
            # Recovery 路径：silent=True 避免在仍未拿到 session 时再 reply "已开启 X 编程模式"，
            # 否则用户会先后看到 "已开启..." 和 "会话启动失败" 两条消息，把启动失败的根因
            # 误导为"已经在模式中但启动不了"。最终统一由下方 mode_session_fail_msg 给出错误。
            logger.info(
                "[%s] handle_message: session missing for chat=%s project=%s thread=%s, "
                "calling enter_mode(silent=True) to (re)start",
                self.mode_name, chat_id[:12] if chat_id else "?",
                project_id or "-", (thread_id or "-")[:12],
            )
            self.enter_mode(message_id, chat_id, silent=True, project=project, thread_id=thread_id)
            if not project:
                working_dir = self.get_working_dir(chat_id)
                try:
                    project, _ = self.project_manager.get_or_create_project_for_path(working_dir, chat_id)
                    project_id = project.project_id
                except Exception:
                    active_project = self.project_manager.get_active_project(chat_id)
                    if active_project:
                        project = active_project
                        project_id = active_project.project_id
            session = self._get_session_manager().get_session(chat_id, project_id=project_id, thread_id=thread_id)
            if not session and thread_id:
                active_project = self.project_manager.get_active_project(chat_id)
                if active_project and active_project is not project:
                    project = active_project
                    project_id = active_project.project_id
                    session = self._get_session_manager().get_session(
                        chat_id,
                        project_id=project_id,
                        thread_id=thread_id,
                    )
            if not session:
                logger.warning(
                    "[%s] handle_message: session still missing after enter_mode recovery; "
                    "chat=%s project=%s thread=%s. Likely ACP startup failed earlier — "
                    "check the previous '[%s] enter_mode session startup failed' log for root cause.",
                    self.mode_name, chat_id[:12] if chat_id else "?",
                    project_id or "-", (thread_id or "-")[:12], self.mode_name,
                )
                self.reply_text(
                    message_id,
                    fmt.format_warning(
                        UI_TEXT["mode_session_fail_msg"].format(name=self.mode_name, cmd=self.mode_name.lower())
                    ),
                    reply_in_thread=True if thread_id else None,
                )
                return

        raw_task_text = text
        text = self.inject_bridge_context(text, project, chat_id=chat_id)
        global_working_dir = self.get_working_dir(chat_id)
        cwd = project.root_path if project else global_working_dir

        # Repo lock: acquire before prompt execution, release after streaming.
        # The lock is held across the entire streaming phase; periodic touch()
        # in the on_event callback keeps last_active_time fresh to prevent
        # idle-timeout release.
        root_path = getattr(project, "root_path", None) if project else None

        from ...repo_lock import LockConflictError

        repo_lock_mgr = None
        needs_release = False
        try:
            _, repo_lock_mgr, needs_release = self._acquire_repo_lock(root_path, chat_id)
        except LockConflictError:
            self.reply_text(
                message_id,
                fmt.format_warning(
                    "仓库当前被其他任务占用，本次任务已安全终止。"
                ),
            )
            return

        try:
            self.handle_response(
                message_id,
                chat_id,
                text,
                session,
                project,
                cwd,
                global_working_dir,
                _repo_lock_mgr=repo_lock_mgr,
                _root_path=root_path,
                _finalization_task_text=raw_task_text,
            )
        finally:
            if needs_release:
                self._release_repo_lock(root_path, chat_id, repo_lock_mgr)

    # ------------------------------------------------------------------
    # handle_response (streaming / non-streaming)
    # ------------------------------------------------------------------
    def handle_response(
        self, message_id: str, chat_id: str, text: str, session: SyncSession, project, cwd: str, global_working_dir: str,
        *, _repo_lock_mgr=None, _root_path: str | None = None,
        _finalization_task_text: str | None = None,
    ):
        from ...card.delivery.channel_client import LarkChannelCardAPIClient
        from ...card.delivery.factory import create_card_delivery
        from ...card.programming_adapter import build_programming_metadata

        project_name = project.project_name if project else None
        project_path = project.root_path if project else global_working_dir
        project_id = project.project_id if project else None

        logger.info("开始 %s 输出: project=%s, path=%s", self.mode_name, project_name, project_path)
        model_name = self._get_model_name_override(project)

        metadata = build_programming_metadata(
            self.mode_name,
            model_name=model_name,
            project_name=project_name,
            working_dir=project_path,
        )

        channel_client_factory = self.ctx.channel_client_factory
        if not callable(channel_client_factory):
            logger.warning("lark-channel 客户端不可用，回退到非流式文本输出")
            self._handle_response_non_streaming(
                message_id, chat_id, text, session, project, global_working_dir,
                _repo_lock_mgr=_repo_lock_mgr, _root_path=_root_path,
                _finalization_task_text=_finalization_task_text,
            )
            return

        try:
            api_client = LarkChannelCardAPIClient(
                channel_client_factory(),
                preallocate_cards=True,
                default_reply_in_thread=self.settings.default_reply_mode == "thread",
                outbound_audit=self.ctx.main_bot_outbound_audit,
                outbound_audit_failure=self.ctx.main_bot_outbound_audit_failure,
                tenant_key_resolver=self.ctx.tenant_key_resolver,
                outbound_target_aliases=lambda target: self._reply_audit_aliases(
                    self._resolve_origin(target)
                ),
            )
        except Exception as exc:
            logger.warning(
                "初始化 lark-channel 卡片客户端失败，回退到非流式文本输出: %s",
                str(exc),
            )
            self._handle_response_non_streaming(
                message_id, chat_id, text, session, project, global_working_dir,
                _repo_lock_mgr=_repo_lock_mgr, _root_path=_root_path,
                _finalization_task_text=_finalization_task_text,
            )
            return

        delivery = create_card_delivery(
            api_client,
            payload_transform=lambda target_chat_id, payload: (
                self._bind_managed_card_revisions(
                    payload,
                    chat_id=target_chat_id,
                )
            ),
            trust_revision_provider=self._managed_card_trust_revisions,
        )
        try:
            delivery_timeout = max(
                2.0,
                2.0 * float(self.settings.card.delivery_api_timeout) + 2.0,
            )
        except Exception:
            delivery_timeout = 12.0

        fallback_to_text = False
        try:
            fallback_to_text = self._handle_response_with_delivery(
                message_id=message_id,
                chat_id=chat_id,
                text=text,
                session=session,
                project=project,
                cwd=cwd,
                global_working_dir=global_working_dir,
                metadata=metadata,
                delivery=delivery,
                delivery_timeout=delivery_timeout,
                project_id=project_id,
                _repo_lock_mgr=_repo_lock_mgr,
                _root_path=_root_path,
                _finalization_task_text=_finalization_task_text,
            )
        finally:
            self._shutdown_owned_programming_delivery(
                delivery,
                timeout=delivery_timeout,
            )
        if fallback_to_text:
            self._handle_response_non_streaming(
                message_id,
                chat_id,
                text,
                session,
                project,
                global_working_dir,
                _repo_lock_mgr=_repo_lock_mgr,
                _root_path=_root_path,
                _finalization_task_text=_finalization_task_text,
            )

    def _handle_response_with_delivery(
        self,
        *,
        message_id: str,
        chat_id: str,
        text: str,
        session: SyncSession,
        project,
        cwd: str,
        global_working_dir: str,
        metadata,
        delivery,
        delivery_timeout: float,
        project_id: str | None,
        _repo_lock_mgr=None,
        _root_path: str | None = None,
        _finalization_task_text: str | None = None,
    ) -> bool:
        """Run one streaming response against a task-owned delivery engine."""
        from ...card.programming_adapter import ProgrammingCardSession
        from ...card.session import CardSession
        from ...card.session.config import SessionConfig

        card_callbacks = build_programming_session_callbacks(
            reply_text_fn=self.reply_text,
            add_reaction=self.add_reaction,
            message_id=message_id,
            chat_id=chat_id,
        )

        def _create_programming_card_session(
            session_metadata,
        ) -> CardSession:
            return CardSession(
                chat_id=chat_id,
                config=SessionConfig(
                    metadata=session_metadata,
                    reply_to=message_id,
                ),
                delivery=delivery,
                callbacks=card_callbacks,
            )

        card_session = _create_programming_card_session(metadata)

        prog_session = ProgrammingCardSession(
            card_session,
            base_metadata=metadata,
            image_uploader=self.upload_acp_image,
            session_factory=_create_programming_card_session,
            continuation_visibility_timeout=delivery_timeout,
        )

        try:
            prog_session.start()
        except Exception as e:
            logger.warning("创建流式卡片失败: %s", str(e))
            prog_session.abort()
            return True

        if not prog_session.wait_until_visible(delivery_timeout):
            logger.warning("首张 lark-channel 编程卡片未成功投递，回退到非流式文本输出")
            prog_session.abort()
            return True

        card_message_id = prog_session.get_message_id()
        if card_message_id:
            try:
                rid = self.ensure_request_id(message_id, chat_id=chat_id, project_id=project_id)
                self.ctx.message_linker.register_origin(
                    message_id, request_id=rid, chat_id=chat_id, project_id=project_id
                )
                self.ctx.message_linker.link_reply(message_id, card_message_id)
            except Exception as e:
                logger.debug("link消息失败(programming): %s", e)

        self._execute_programming_response(
            message_id,
            chat_id,
            text,
            session,
            project,
            cwd,
            global_working_dir,
            prog_session=prog_session,
            card_message_id=card_message_id,
            delivery_timeout=delivery_timeout,
            _repo_lock_mgr=_repo_lock_mgr,
            _root_path=_root_path,
            _finalization_task_text=_finalization_task_text,
        )
        return False

    @staticmethod
    def _shutdown_owned_programming_delivery(delivery, *, timeout: float) -> None:
        """Best-effort cleanup for a per-task Channel delivery engine."""
        try:
            if not delivery.shutdown(timeout=timeout):
                logger.warning(
                    "lark-channel 编程卡片资源未在 %.1fs 内排空；保留进程级关闭重试",
                    timeout,
                )
        except Exception:
            logger.exception("关闭 lark-channel 编程卡片资源失败")

    def _execute_programming_response(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        session: SyncSession,
        project,
        cwd: str,
        global_working_dir: str,
        *,
        prog_session=None,
        card_message_id: str | None = None,
        delivery_timeout: float = 12.0,
        _repo_lock_mgr=None,
        _root_path: str | None = None,
        _finalization_task_text: str | None = None,
    ) -> None:
        """Run one automatic ACP task, independent of its output transport."""
        from ...acp.models import ACPEventType
        from ...repo_lock import get_repo_lock_heartbeat_interval
        from ...tasking import get_current_task_run_id
        from ...thread import get_current_thread_id
        from ...utils.heartbeat import RepoLockHeartbeat

        timeout = self.settings.coco_execution_timeout if self.is_coco else self.settings.claude_execution_timeout
        update_count = [0]
        touch_interval = get_repo_lock_heartbeat_interval(_repo_lock_mgr)
        _repo_lock_owner_id = get_current_task_run_id()
        heartbeat_stop = threading.Event()
        if _repo_lock_mgr and _root_path:
            heartbeat = RepoLockHeartbeat(
                heartbeat_stop,
                lambda: _repo_lock_mgr.touch(
                    _root_path,
                    chat_id,
                    owner_id=_repo_lock_owner_id,
                ),
                interval=touch_interval,
                name=f"prog-{_root_path}",
            )
            heartbeat.start()
        else:
            heartbeat = None

        renderer = ACPEventRenderer() if prog_session is None else None
        image_keys: list[str] = []
        seen_image_ids: set[str] = set()
        image_failures = [0]
        child_lifecycle_proof = AuthoritativeChildLifecycleProof()

        def on_event(event) -> None:
            update_count[0] += 1
            child_lifecycle_proof.observe_event(event)
            if prog_session is not None:
                try:
                    prog_session.on_event(event)
                except Exception as exc:
                    logger.warning(
                        "card session event处理失败: %s",
                        str(exc),
                        exc_info=True,
                    )
                return
            if event.event_type is ACPEventType.IMAGE_CHUNK:
                image = event.image
                if image is None or image.image_id in seen_image_ids:
                    return
                seen_image_ids.add(image.image_id)
                image_key = self.upload_acp_image(image)
                if image_key:
                    image_keys.append(image_key)
                else:
                    image_failures[0] += 1
                return
            renderer.process_event(event)

        final_response = ""
        prompt_outcome = PromptOutcome.INCOMPLETE
        prompt_stop_reason = "exception"
        terminal_notice = ""
        terminal_exception: Exception | None = None
        active_session = [session]
        entered_finalization = [False]
        retired_session_objects: set[int] = set()
        thread_id = get_current_thread_id()

        def _start_finalization() -> None:
            entered_finalization[0] = True
            logger.warning(
                "%s ACP任务进入安全收尾阶段: total_timeout=%ss reserve=%ss",
                self.mode_name,
                timeout,
                _configured_finalization_reserve(self.settings),
            )

        def _replace_dead_session(startup_budget_s: float) -> SyncSession:
            replacement = self._replace_timed_out_session(
                chat_id=chat_id,
                project=project,
                cwd=cwd,
                thread_id=thread_id,
                timed_out_session=active_session[0],
                startup_budget_s=startup_budget_s,
            )
            active_session[0] = replacement
            return replacement

        def _retire_session(
            active: SyncSession,
            retirement_budget_s: float = 0.001,
        ) -> None:
            session_object_id = id(active)
            if session_object_id in retired_session_objects:
                return
            try:
                setattr(active, "_force_dead", True)
            except Exception:
                logger.warning("failed to poison-mark timed-out ACP session")
            self._retire_finalization_session(
                chat_id=chat_id,
                project=project,
                thread_id=thread_id,
                active_session=active,
                retirement_budget_s=retirement_budget_s,
            )
            retired_session_objects.add(session_object_id)

        def _execute_window(
            window_session: SyncSession,
            window_prompt: str,
        ):
            active_session[0] = window_session
            return run_prompt_with_continuation(
                window_session,
                window_prompt,
                on_event=on_event,
                timeout_s=timeout,
                finalization_reserve_s=_configured_finalization_reserve(
                    self.settings
                ),
                finalization_task_text=_finalization_task_text,
                on_finalization_start=_start_finalization,
                on_continuation_start=(
                    prog_session.begin_continuation_turn
                    if prog_session is not None
                    else None
                ),
                replace_dead_session=_replace_dead_session,
                retire_finalization_session=_retire_session,
            )

        def _resume_window(
            _retired_session: SyncSession,
            provider_session_id: str,
        ) -> SyncSession:
            project_id = project.project_id if project else None
            resume_timeout = max(
                60.0,
                float(getattr(self.settings, "acp_startup_timeout", 60) or 60),
            )
            resumed = self._get_session_manager().resume_retired_session(
                chat_id,
                cwd=cwd,
                session_id=provider_session_id,
                startup_timeout=resume_timeout,
                project_id=project_id,
                agent_type_override=self._get_agent_type_override(project),
                model_name=self._get_model_name_override(project),
                thread_id=thread_id,
            )
            active_session[0] = resumed
            if thread_id and project:
                self._register_thread_context(
                    thread_id,
                    chat_id,
                    project,
                    resumed,
                )
            return resumed

        def _on_window_rollover(
            next_window: int,
            max_windows: int,
        ) -> None:
            logger.warning(
                "%s ACP任务自动续开执行窗口: window=%d/%d",
                self.mode_name,
                next_window,
                max_windows,
            )
            if prog_session is not None:
                prog_session.begin_continuation_turn()

        try:
            execution = run_prompt_across_execution_windows(
                session,
                text,
                task_scope=_finalization_task_text,
                max_windows=_configured_execution_windows(self.settings),
                execute_window=_execute_window,
                resume_window=_resume_window,
                on_window_rollover=_on_window_rollover,
            )
            result = execution.result
            assessment = execution.assessment
            _log_prompt_execution(self.mode_name, execution)
            prompt_outcome = assessment.outcome
            prompt_stop_reason = assessment.stop_reason
            reconciled_stale_children = _has_reconciled_stale_child_history(
                execution,
                child_lifecycle_proof,
            )
            if reconciled_stale_children:
                prompt_outcome = PromptOutcome.COMPLETED
                logger.warning(
                    "%s ACP历史子代理状态已由最终权威快照完成对账: "
                    "stale_child_records=%d",
                    self.mode_name,
                    assessment.unresolved_child_tool_calls,
                )
            result_text = (getattr(result, "text", None) or "").strip()
            streamed_response = (
                prog_session.get_final_text()
                if prog_session is not None
                else renderer.get_final_content()
            )
            response_text = (
                streamed_response or result_text
                if prog_session is not None
                else result_text or streamed_response
            )
            if prompt_outcome is PromptOutcome.COMPLETED:
                if prog_session is not None and not streamed_response and result_text:
                    prog_session.on_text(result_text)
                final_response = response_text or UI_TEXT["mode_exec_complete"]
                if prog_session is not None:
                    if reconciled_stale_children:
                        prog_session.finish(
                            warning=UI_TEXT[
                                "mode_exec_child_history_reconciled"
                            ].format(
                                count=assessment.unresolved_child_tool_calls,
                            ),
                            details=_execution_diagnostic_details(execution),
                            detail_action=_diagnostic_action(
                                chat_id,
                                message_id,
                            ),
                        )
                    else:
                        prog_session.finish()
            elif prompt_outcome is PromptOutcome.CANCELLED:
                terminal_notice = UI_TEXT["mode_exec_cancelled_msg"].format(
                    reason=assessment.detail,
                )
                final_response = _append_execution_notice(
                    response_text,
                    terminal_notice,
                )
                if prog_session is not None:
                    prog_session.cancel(reason=assessment.stop_reason)
            else:
                terminal_notice = _incomplete_notice(execution)
                final_response = _append_execution_notice(
                    response_text,
                    terminal_notice,
                )
                if prog_session is not None:
                    prog_session.fail(
                        terminal_notice,
                        unfinished_subagent_status=(
                            "cancelled"
                            if execution.entered_finalization
                            else "failed"
                        ),
                        details=_execution_diagnostic_details(execution),
                        detail_action=_diagnostic_action(chat_id, message_id),
                    )
        except TimeoutError as e:
            terminal_exception = e
            try:
                _retire_session(active_session[0])
            except Exception:
                logger.error(
                    "%s ACP超时会话退休失败；会话已标记为不可复用",
                    self.mode_name,
                    exc_info=True,
                )
            prompt_stop_reason = "timeout"
            terminal_notice = UI_TEXT["mode_exec_timeout_msg"].format(
                error=get_error_detail(e)
            )
            final_response = _append_execution_notice(
                prog_session.get_final_text() if prog_session is not None else renderer.get_final_content(),
                terminal_notice,
            )
            log_exception(logger, f"{self.mode_name} ACP执行超时", e, level=logging.WARNING)
            if prog_session is not None:
                prog_session.fail(
                    terminal_notice,
                    unfinished_subagent_status="cancelled",
                    details=f"stop_reason=timeout\nerror={get_error_detail(e)}",
                    detail_action=_diagnostic_action(chat_id, message_id),
                )
        except Exception as e:
            terminal_exception = e
            if (
                entered_finalization[0]
                or getattr(active_session[0], "_force_dead", False) is True
            ) and id(active_session[0]) not in retired_session_objects:
                try:
                    _retire_session(active_session[0])
                except Exception:
                    logger.error(
                        "%s ACP安全收尾会话退休失败",
                        self.mode_name,
                        exc_info=True,
                )
            prompt_stop_reason = "exception"
            terminal_notice = UI_TEXT["mode_exec_exception_msg"].format(
                error=get_error_detail(e)
            )
            final_response = _append_execution_notice(
                prog_session.get_final_text() if prog_session is not None else renderer.get_final_content(),
                terminal_notice,
            )
            log_exception(logger, f"{self.mode_name} ACP执行异常", e)
            if prog_session is not None:
                prog_session.fail(
                    terminal_notice,
                    unfinished_subagent_status=(
                        "cancelled" if entered_finalization[0] else "failed"
                    ),
                    details=(
                        f"stop_reason=exception\n"
                        f"error={get_error_detail(e)}"
                    ),
                    detail_action=_diagnostic_action(chat_id, message_id),
                )
            from ...utils.errors import GhostAPError
            if prog_session is not None and isinstance(e, GhostAPError) and e.quick_actions:
                self.send_error_card(chat_id, e, title=UI_TEXT["mode_exec_exception_title"], origin_message_id=message_id)
        finally:
            heartbeat_stop.set()
            if heartbeat is not None:
                heartbeat.join(timeout=2)
            if prog_session is not None:
                terminal_delivered = (
                    prog_session.wait_delivery_idle(delivery_timeout)
                    and prog_session.terminal_delivery_succeeded()
                )
                if not terminal_delivered:
                    logger.warning("lark-channel 编程卡片终态投递失败，改发文本结果")
                    prog_session.abort()
                    if final_response:
                        self.reply_text(message_id, final_response)

        if prog_session is None:
            if terminal_exception is not None:
                title = (
                    UI_TEXT["mode_exec_timeout_title"]
                    if prompt_stop_reason == "timeout"
                    else UI_TEXT["mode_exec_exception_title"]
                )
                _msg_type, content = SystemBuilder.build_error_card(
                    terminal_exception,
                    title=title,
                    project=project,
                )
                self.reply_card(message_id, content)
            else:
                title_key = {
                    PromptOutcome.COMPLETED: "mode_exec_complete",
                    PromptOutcome.CANCELLED: "mode_exec_cancelled_title",
                    PromptOutcome.INCOMPLETE: "mode_exec_incomplete_title",
                }[prompt_outcome]
                response_with_dir = (
                    f"{final_response}\n\n---\n"
                    f"{UI_TEXT['mode_working_dir_label'].format(path=global_working_dir)}"
                )
                if image_failures[0]:
                    response_with_dir += "\n\n🖼️ 部分图片产物暂时无法展示。"
                if image_keys:
                    _msg_type, content = ProjectBuilder.build_project_response_card(
                        project,
                        f"{self.mode_name} · {UI_TEXT[title_key]}",
                        response_with_dir,
                        show_buttons=bool(project),
                        image_keys=image_keys,
                    )
                    self.reply_card(message_id, content)
                else:
                    self.reply_text(message_id, response_with_dir)
                if prompt_outcome is PromptOutcome.COMPLETED:
                    self.add_reaction(message_id, EmojiHook.SUCCESS_EMOJI_DEFAULT)

        logger.info(
            "%s ACP输出结束: outcome=%s, stop_reason=%s, 事件数=%d, 最终长度=%d",
            self.mode_name,
            prompt_outcome.value,
            prompt_stop_reason,
            update_count[0],
            len(final_response),
        )

        try:
            if project:
                final_session = active_session[0]
                if (
                    id(final_session) in retired_session_objects
                    or getattr(final_session, "_force_dead", False) is True
                ):
                    self._clear_snapshot_for_session(project, session)
                    self._clear_snapshot_for_session(project, final_session)
                else:
                    self._update_snapshot_on_project(
                        project,
                        text,
                        final_session.message_count,
                        final_session.session_id,
                    )
                project.add_conversation("user", text, message_id)
                project.add_conversation("assistant", final_response)
                source = self.mode_name.lower()
                self.context_manager.update_context(
                    project.project_id,
                    conversation={"role": "user", "content": text, "source_mode": source, "message_id": message_id},
                    chat_id=chat_id,
                )
                self.context_manager.update_context(
                    project.project_id,
                    conversation={"role": "assistant", "content": final_response, "source_mode": source},
                    chat_id=chat_id,
                )
        except Exception as e:
            logger.warning("编程后处理异常(不影响表情回复): %s", e, exc_info=True)

        if card_message_id and project:
            self.register_message_project(card_message_id, project)

    def _handle_response_non_streaming(
        self, message_id: str, chat_id: str, text: str, session: SyncSession, project, global_working_dir: str,
        *, _repo_lock_mgr=None, _root_path: str | None = None,
        _finalization_task_text: str | None = None,
    ):
        """Run the same automatic pipeline through the text/card fallback."""
        cwd = project.root_path if project else global_working_dir
        self._execute_programming_response(
            message_id,
            chat_id,
            text,
            session,
            project,
            cwd,
            global_working_dir,
            _repo_lock_mgr=_repo_lock_mgr,
            _root_path=_root_path,
            _finalization_task_text=_finalization_task_text,
        )

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
    def handle_card_enter(self, message_id: str, chat_id: str, project_id: str, value: Optional[dict] = None):
        if project_id:
            project = self.project_manager.get_project_for_chat(project_id, chat_id)
            if project:
                self.project_manager.set_active_project(chat_id, project_id)
                self.enter_mode(message_id, chat_id, project=project)
                return

        self.enter_mode(message_id, chat_id)

    def handle_card_exit(self, message_id: str, chat_id: str, project_id: str, value: Optional[dict] = None):
        from ...thread import get_current_thread_id
        if project_id:
            project = self.project_manager.get_project_for_chat(project_id, chat_id)
            if project and not get_current_thread_id():
                self._set_mode_on_project(project, False)
            self.exit_mode(message_id, chat_id, project=project)
            return
        self.exit_mode(message_id, chat_id)

# ======================================================================
# Concrete subclasses
# ======================================================================


class CocoModeHandler(ProgrammingModeHandler):
    mode_name = "Coco"
    mode_emoji = "🤖"
    is_coco = True
    interaction_mode = InteractionMode.COCO
    mode_key = "coco"
    context_source = ContextSourceMode.COCO
    thinking_text = UI_TEXT["mode_thinking_msg"].format(emoji="🤔", name="Coco")


class ClaudeModeHandler(ProgrammingModeHandler):
    mode_name = "Claude"
    mode_emoji = "🔮"
    interaction_mode = InteractionMode.CLAUDE
    mode_key = "claude"
    context_source = ContextSourceMode.CLAUDE
    thinking_text = UI_TEXT["mode_thinking_msg"].format(emoji="🔮", name="Claude")

    def _uses_claude_cli(self) -> bool:
        return True


class AidenModeHandler(ProgrammingModeHandler):
    mode_name = "Aiden"
    mode_emoji = "🎯"
    interaction_mode = InteractionMode.AIDEN
    mode_key = "aiden"
    context_source = ContextSourceMode.AIDEN
    thinking_text = UI_TEXT["mode_thinking_msg"].format(emoji="🎯", name="Aiden")


class CodexModeHandler(ProgrammingModeHandler):
    mode_name = "Codex"
    mode_emoji = "⚡"
    interaction_mode = InteractionMode.CODEX
    mode_key = "codex"
    context_source = ContextSourceMode.CODEX
    thinking_text = UI_TEXT["mode_thinking_msg"].format(emoji="⚡", name="Codex")


class GeminiModeHandler(ProgrammingModeHandler):
    mode_name = "Gemini"
    mode_emoji = "✨"
    interaction_mode = InteractionMode.GEMINI
    mode_key = "gemini"
    context_source = ContextSourceMode.GEMINI
    thinking_text = UI_TEXT["mode_thinking_msg"].format(emoji="✨", name="Gemini")


class TraexModeHandler(ProgrammingModeHandler):
    mode_name = "Traex"
    mode_emoji = "🚀"
    interaction_mode = InteractionMode.TRAEX
    mode_key = "traex"
    context_source = ContextSourceMode.TRAEX
    thinking_text = UI_TEXT["mode_thinking_msg"].format(emoji="🚀", name="Traex")


class GrokModeHandler(ProgrammingModeHandler):
    mode_name = "Grok"
    mode_emoji = "🌌"
    interaction_mode = InteractionMode.GROK
    mode_key = "grok"
    context_source = ContextSourceMode.GROK
    thinking_text = UI_TEXT["mode_thinking_msg"].format(emoji="🌌", name="Grok")
