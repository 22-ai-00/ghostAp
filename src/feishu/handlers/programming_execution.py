"""Execution pipeline mixin for programming mode handlers."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from ...card.ui_text import UI_TEXT
from ...project import ContextSourceMode
from ...utils.errors import get_error_detail, log_exception
from ..emoji import EmojiReaction
from ..message_formatter import FeishuMessageFormatter as fmt
from .programming_support import (
    _append_execution_notice,
    _configured_execution_windows,
    _configured_finalization_reserve,
    _configured_programming_idle_timeout,
    _diagnostic_action,
    _execution_diagnostic_details,
    _format_swarm_completion_notice,
    _has_reconciled_stale_child_history,
    _incomplete_notice,
    _log_final_unresolved_tool_trace,
    _log_prompt_execution,
    _was_already_streamed_on_main,
    build_programming_session_callbacks,
)

if TYPE_CHECKING:
    from ...agent_session import SyncSession
    from ...project import ProjectContext

logger = logging.getLogger("src.feishu.handlers.programming")

_PROGRAMMING_COT_REQUEST_TIMEOUT_S = 5.0
_SAFE_FEISHU_APP_ID = re.compile(r"cli_[A-Za-z0-9_-]{1,256}\Z")


def _programming_module():
    """Return the programming handler module for runtime-patchable symbols."""
    from . import programming as _prog_mod

    return _prog_mod


class ProgrammingExecutionMixin:
    """Mixin providing message handling, card streaming, COT, and ACP execution."""

    # ------------------------------------------------------------------
    # handle_message
    # ------------------------------------------------------------------
    def handle_message(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        project: Optional["ProjectContext"] = None,
        *,
        expected_session: Optional[SyncSession] = None,
    ):
        from ...thread import get_current_thread_id

        project_id = project.project_id if project else None
        thread_id = get_current_thread_id()
        session = self._get_session_manager().get_session(chat_id, project_id=project_id, thread_id=thread_id)

        if expected_session is not None and session is not expected_session:
            # A model activation dispatches its pending prompt only to the exact
            # session it committed.  If /exit or a newer activation replaced that
            # session in the narrow hand-off window, do not recover/re-enter it.
            logger.info(
                "[%s] pending prompt skipped because the committed ACP session "
                "is no longer current: chat=%s project=%s thread=%s",
                self.mode_name,
                chat_id[:12] if chat_id else "?",
                project_id or "-",
                (thread_id or "-")[:12],
            )
            return

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
        try:
            text = self.inject_bridge_context(
                text,
                project,
                chat_id=chat_id,
                thread_id=thread_id or "",
            )
        except TypeError:
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
        active_run_key = None
        active_run = None
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
            active_run_key, active_run = self._register_active_programming_run(
                chat_id,
                project,
                task_text=raw_task_text,
                cwd=cwd,
            )
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
            if active_run_key is not None and active_run is not None:
                self._unregister_active_programming_run(
                    active_run_key,
                    active_run,
                )
            if needs_release:
                self._release_repo_lock(root_path, chat_id, repo_lock_mgr)

    # ------------------------------------------------------------------
    # handle_response (streaming / non-streaming)
    # ------------------------------------------------------------------
    def _notify_programming_text_fallback(self, message_id: str) -> None:
        """Make a transport downgrade visible before the blocking ACP turn."""

        notice = UI_TEXT["mode_stream_fallback_running"]
        app_id = str(getattr(self.settings, "app_id", "") or "").strip()
        if _SAFE_FEISHU_APP_ID.fullmatch(app_id) is not None:
            permission_url = (
                f"https://open.feishu.cn/app/{app_id}/auth"
                "?q=cardkit:card:write&op_from=openapi&token_type=tenant"
            )
            notice = (
                f"{notice}\n"
                f"{UI_TEXT['mode_stream_fallback_permission_hint']}\n"
                f"{permission_url}"
            )
        self.reply_text(message_id, notice)

    def _create_programming_process_sink(
        self,
        *,
        message_id: str,
        chat_id: str,
        input_text: str,
        on_segment_started: Callable[[str], None] | None = None,
    ):
        """Build the optional Feishu COT sink without starting network I/O."""
        if getattr(self.settings, "feishu_cot_enabled", None) is not True:
            return None
        try:
            from ..cot import FeishuCOTAPIClient, FeishuCOTStream

            api = FeishuCOTAPIClient(
                self.ctx.api_client_factory(),
                outbound_audit=self.ctx.main_bot_outbound_audit,
                outbound_audit_failure=self.ctx.main_bot_outbound_audit_failure,
                tenant_key_resolver=self.ctx.tenant_key_resolver,
                outbound_target_aliases=lambda target: self._reply_audit_aliases(
                    self._resolve_origin(target)
                ),
                trust_revision_provider=self._managed_card_trust_revisions,
            )
            return FeishuCOTStream(
                api,
                chat_id=chat_id,
                origin_message_id=message_id,
                reply_in_thread=self.settings.default_reply_mode == "thread",
                input_text=input_text,
                detail=self.settings.feishu_cot_detail,
                request_timeout=_PROGRAMMING_COT_REQUEST_TIMEOUT_S,
                on_segment_started=on_segment_started,
            )
        except Exception:
            logger.warning(
                "初始化飞书 COT 过程消息失败，继续使用流式卡片",
                exc_info=True,
            )
            return None

    def _link_process_segment(
        self,
        *,
        message_id: str,
        chat_id: str,
        project_id: str | None,
        project: Optional["ProjectContext"],
        process_message_id: str,
        linked_process_segments: set[str],
    ) -> None:
        if process_message_id in linked_process_segments:
            return
        linked_process_segments.add(process_message_id)
        try:
            rid = self.ensure_request_id(
                message_id,
                chat_id=chat_id,
                project_id=project_id,
            )
            self.ctx.message_linker.register_origin(
                message_id,
                request_id=rid,
                chat_id=chat_id,
                project_id=project_id,
            )
            self.ctx.message_linker.link_reply(
                message_id,
                process_message_id,
            )
        except Exception as exc:
            logger.debug("link消息失败(programming COT segment): %s", exc)
        if project:
            try:
                self.register_message_project(process_message_id, project)
            except Exception as exc:
                logger.debug("项目消息映射失败(programming COT segment): %s", exc)

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
        try:
            model_name = self._get_model_name_override(project, chat_id=chat_id)
        except TypeError:
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
            self._notify_programming_text_fallback(message_id)
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
            self._notify_programming_text_fallback(message_id)
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
        linked_process_segments: set[str] = set()

        process_sink = self._create_programming_process_sink(
            message_id=message_id,
            chat_id=chat_id,
            input_text=_finalization_task_text or text,
            on_segment_started=lambda process_message_id: self._link_process_segment(
                message_id=message_id,
                chat_id=chat_id,
                project_id=project_id,
                project=project,
                process_message_id=process_message_id,
                linked_process_segments=linked_process_segments,
            ),
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
                process_sink=process_sink,
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
            self._notify_programming_text_fallback(message_id)
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
        process_sink=None,
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
            process_sink=process_sink,
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
        origin_registered = False
        if card_message_id:
            try:
                rid = self.ensure_request_id(message_id, chat_id=chat_id, project_id=project_id)
                self.ctx.message_linker.register_origin(
                    message_id, request_id=rid, chat_id=chat_id, project_id=project_id
                )
                origin_registered = True
                self.ctx.message_linker.link_reply(message_id, card_message_id)
            except Exception as e:
                logger.debug("link消息失败(programming): %s", e)

        process_message_id = None
        if process_sink is not None and prog_session.activate_process_sink():
            process_message_id = prog_session.get_process_message_id()
            if process_message_id:
                try:
                    if not origin_registered:
                        rid = self.ensure_request_id(
                            message_id,
                            chat_id=chat_id,
                            project_id=project_id,
                        )
                        self.ctx.message_linker.register_origin(
                            message_id,
                            request_id=rid,
                            chat_id=chat_id,
                            project_id=project_id,
                        )
                    self.ctx.message_linker.link_reply(
                        message_id,
                        process_message_id,
                    )
                except Exception as e:
                    logger.debug("link消息失败(programming COT): %s", e)
                if project:
                    try:
                        self.register_message_project(process_message_id, project)
                    except Exception as e:
                        logger.debug("项目消息映射失败(programming COT): %s", e)

        try:
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
        except BaseException:
            # COT is activated before the ACP execution helper initializes its
            # own terminal guards.  Fence both transports if that setup raises
            # so an already-created native run cannot remain live indefinitely.
            prog_session.abort()
            raise
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
        from ...acp import ACPEventRenderer
        from ...acp.models import (
            ACPEventType,
            TerminalOutputProvenance,
            resolve_prompt_terminal_output,
        )
        from ...acp.native_commands import parse_slash_command
        from ...acp.outcome import PromptOutcome
        from ...acp.swarm_lifecycle import is_swarm_working_event
        from ...card.builders.project import ProjectBuilder
        from ...card.builders.system import SystemBuilder
        from ...card.hooks import EmojiHook
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
        mod = _programming_module()
        child_lifecycle_proof = getattr(mod, "AuthoritativeChildLifecycleProof")()
        swarm_working_notified = [False]
        swarm_completed_flag = [False]
        active_run = self._get_active_programming_run(chat_id, project)
        if active_run is not None:
            active_run.session = session
            active_run.prog_session = prog_session

        def on_event(event) -> None:
            update_count[0] += 1
            child_lifecycle_proof.observe_event(event)
            if not swarm_working_notified[0] and is_swarm_working_event(event):
                swarm_working_notified[0] = True
                try:
                    self.add_reaction(message_id, EmojiReaction.on_swarm_working())
                except Exception as exc:
                    logger.warning(
                        "添加 Swarm 工作中表情回复异常: %s",
                        exc,
                        exc_info=True,
                    )
            if active_run is not None:
                active_run.observe(event)
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

        result = None
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
            if active_run is not None:
                active_run.session = replacement
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
            if active_run is not None:
                active_run.session = window_session
            return getattr(mod, "run_prompt_with_continuation")(
                window_session,
                window_prompt,
                on_event=on_event,
                timeout_s=timeout,
                finalization_reserve_s=_configured_finalization_reserve(
                    self.settings
                ),
                idle_timeout_s=_configured_programming_idle_timeout(
                    self.settings
                ),
                finalization_task_text=_finalization_task_text,
                on_finalization_start=_start_finalization,
                on_continuation_start=(
                    prog_session.begin_continuation_turn
                    if prog_session is not None
                    else None
                ),
                on_child_reconciliation_start=(
                    getattr(
                        prog_session,
                        "freeze_result_conclusion_fallback",
                        None,
                    )
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
                model_name=self._session_bound_model_name(
                    _retired_session,
                    self._get_model_name_override(project, chat_id=chat_id),
                ),
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
            execution = getattr(mod, "run_prompt_across_execution_windows")(
                session,
                text,
                task_scope=_finalization_task_text,
                max_windows=_configured_execution_windows(self.settings),
                execute_window=_execute_window,
                resume_window=_resume_window,
                on_window_rollover=_on_window_rollover,
            )
            result = execution.result
            swarm_completed_flag[0] = bool(
                swarm_working_notified[0] or getattr(result, "swarm_completed", False)
            )
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
            if (
                prompt_outcome is PromptOutcome.INCOMPLETE
                and assessment.incomplete_tool_calls > 0
            ):
                _log_final_unresolved_tool_trace(self.mode_name, assessment)
            terminal_text, terminal_provenance = resolve_prompt_terminal_output(
                result
            )
            result_text = (getattr(result, "text", None) or "").strip()
            streamed_response = (
                prog_session.get_final_text()
                if prog_session is not None
                else renderer.get_final_content()
            )
            if prog_session is not None:
                has_trusted_fn = getattr(
                    prog_session, "has_trusted_final_answer", None
                )
                has_fallback_fn = getattr(
                    prog_session, "has_result_conclusion_fallback", None
                )
                has_locked_conclusion = (
                    (callable(has_trusted_fn) and has_trusted_fn() is True)
                    or (callable(has_fallback_fn) and has_fallback_fn() is True)
                )
                high_authority_result = bool(
                    terminal_text
                    and (
                        terminal_provenance
                        in {
                            TerminalOutputProvenance.SWARM_FINAL_TURN,
                            TerminalOutputProvenance.FINAL_ANSWER_PHASE,
                        }
                        or (
                            terminal_provenance
                            is TerminalOutputProvenance.UNSTREAMED_RESPONSE
                            and not _was_already_streamed_on_main(
                                terminal_text,
                                streamed_response,
                            )
                        )
                    )
                )
                if has_locked_conclusion:
                    response_text = (
                        prog_session.get_conclusion_text().strip()
                        or streamed_response
                    )
                elif high_authority_result:
                    prog_session.set_result_conclusion_fallback(terminal_text)
                    response_text = terminal_text
                else:
                    conclusion_text = prog_session.get_conclusion_text().strip()
                    if conclusion_text:
                        response_text = conclusion_text
                    elif (
                        terminal_provenance is not TerminalOutputProvenance.NONE
                        and terminal_text
                    ):
                        response_text = terminal_text
                    else:
                        response_text = streamed_response
            else:
                has_locked_conclusion = False
                high_authority_result = False
                response_text = terminal_text or streamed_response
            if prompt_outcome is PromptOutcome.COMPLETED:
                parsed_native = parse_slash_command((text or "").splitlines()[0] if text else "")
                native_summary = ""
                if parsed_native is not None and not streamed_response and not result_text:
                    native_cmd_name = f"/{parsed_native[0]}"
                    tool_call_count = len(getattr(result, "tool_calls", None) or [])
                    if tool_call_count > 0:
                        native_summary = UI_TEXT["mode_native_cmd_complete_with_tools"].format(
                            command=native_cmd_name,
                            tool_count=tool_call_count,
                        )
                    else:
                        native_summary = UI_TEXT["mode_native_cmd_complete"].format(
                            command=native_cmd_name,
                        )
                    usage_info = getattr(result, "usage", None)
                    if usage_info is not None and getattr(usage_info, "size", 0) > 0:
                        pct = int(round((usage_info.used / usage_info.size) * 100))
                        native_summary += f" · 上下文 {usage_info.used:,}/{usage_info.size:,} ({pct}%)"
                if prog_session is not None:
                    if high_authority_result:
                        if (
                            not streamed_response
                            and not prog_session.get_conclusion_text().strip()
                        ):
                            prog_session.on_text(terminal_text)
                    elif not has_locked_conclusion:
                        summary_text = (
                            terminal_text
                            if terminal_provenance is not TerminalOutputProvenance.NONE
                            else ""
                        ) or native_summary
                        if summary_text and not streamed_response:
                            prog_session.set_result_conclusion_fallback(summary_text)
                        if (
                            not streamed_response
                            and summary_text
                            and not prog_session.get_conclusion_text().strip()
                        ):
                            prog_session.on_text(summary_text)
                final_response = response_text or native_summary or UI_TEXT["mode_exec_complete"]
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
                    final_response = (
                        prog_session.get_conclusion_text().strip()
                        or final_response
                    )
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
                (
                    prog_session.get_conclusion_text().strip()
                    or prog_session.get_final_text()
                )
                if prog_session is not None
                else renderer.get_final_content(),
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
                (
                    prog_session.get_conclusion_text().strip()
                    or prog_session.get_final_text()
                )
                if prog_session is not None
                else renderer.get_final_content(),
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
                elif prompt_outcome is PromptOutcome.COMPLETED and swarm_completed_flag[0]:
                    try:
                        self.reply_text(
                            message_id,
                            _format_swarm_completion_notice(final_response),
                        )
                    except Exception as exc:
                        logger.warning(
                            "发送 Swarm 完成通知失败: %s",
                            exc,
                            exc_info=True,
                        )

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
                raw_user_text = _finalization_task_text or text
                if (
                    id(final_session) in retired_session_objects
                    or getattr(final_session, "_force_dead", False) is True
                ):
                    self._clear_snapshot_for_session(project, session, chat_id=chat_id)
                    self._clear_snapshot_for_session(project, final_session, chat_id=chat_id)
                else:
                    self._update_snapshot_on_project(
                        project,
                        raw_user_text,
                        final_session.message_count,
                        final_session.session_id,
                        chat_id=chat_id,
                    )
                source = (
                    self.context_source.value
                    if isinstance(self.context_source, ContextSourceMode)
                    else self.mode_key
                )
                modified_files = sorted(getattr(result, "modified_files", None) or ())
                thread_scope = thread_id or ""
                try:
                    self.context_manager.update_context(
                        project.project_id,
                        conversation={"role": "user", "content": raw_user_text, "source_mode": source, "message_id": message_id},
                        file_changes=modified_files or None,
                        chat_id=chat_id,
                        thread_id=thread_scope,
                    )
                    self.context_manager.update_context(
                        project.project_id,
                        conversation={"role": "assistant", "content": final_response, "source_mode": source},
                        chat_id=chat_id,
                        thread_id=thread_scope,
                    )
                except TypeError:
                    self.context_manager.update_context(
                        project.project_id,
                        conversation={"role": "user", "content": raw_user_text, "source_mode": source, "message_id": message_id},
                        file_changes=modified_files or None,
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
