"""Message parsing, validation, context resolution, and dispatch pipeline for FeishuWSClient."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..trust.models import EffectiveTrust
    from .ws_client import P2ImMessageReceiveV1

from ..access_control import AccessOperation
from ..card.ui_text import UI_TEXT
from ..mode import PROGRAMMING_MODE_VALUES, PROGRAMMING_MODES
from ..project import get_project_acp_tool_name
from ..thread import get_current_thread_id
from ..trust.models import TrustZone
from ..utils.errors import get_error_detail
from .emoji import EmojiReaction
from .handlers import SystemHandler
from .image_handler import FeishuImageHandler
from .slash_command_parser import SlashCommandParser
from .ws_event_router import WSErrorAction, classify_ws_error
from .ws_trust_access import _BOOTSTRAP_ACCESS_OPERATIONS

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("ghostap.audit")


class WSMessagePipelineMixin:
    """Mixin handling async message processing, validation, context, and dispatch."""

    def _extract_text_from_message(self, data: P2ImMessageReceiveV1) -> str:
        """Extract message text with the same parser used by async dispatch."""
        try:
            message = data.event.message
            content_str = message.content
            if not content_str:
                return ""
            parsed = self._get_image_handler().parse_message(
                message.message_type,
                content_str,
            )
            return self._clean_at_text(parsed.text)
        except (AttributeError, KeyError, TypeError, ValueError):
            return ""

    def _build_control_queue_key(self, *, chat_id: str, project_id: Optional[str], text: str) -> Optional[str]:
        """为编程初始化与 spec 命令构造串行控制队列 key。"""
        from .product_catalog import is_programming_entry_command

        normalized = (text or "").strip()
        if not normalized:
            return None
        if not (
            SystemHandler.is_spec_command(normalized)
            or is_programming_entry_command(normalized)
            or SystemHandler.is_workflow_command(normalized)
        ):
            return None
        queue_suffix = project_id or "default"
        return f"{chat_id}:control:{queue_suffix}"

    def _process_message_async(
        self,
        data: P2ImMessageReceiveV1,
        task_ctx=None,
        shell_fast_tracked: bool = False,
        effective_trust: EffectiveTrust | None = None,
        message_reservation_id: str | None = None,
        message_reservation_owner: str | None = None,
    ):
        """消息处理主逻辑（运行在 scheduler 线程池中）。

        大致流程：校验 → 解析文本/图片 → 解析项目上下文 → 路由到对应模式/引擎。
        """
        from ..thread import (
            set_current_is_p2p,
            set_current_mentioned_names,
            set_current_sender_id,
            set_current_sender_name,
            set_current_sender_union_id,
            set_current_tenant_key,
            set_current_thread_id,
        )

        def release_message_reservation() -> None:
            if (
                message_reservation_id is not None
                and message_reservation_owner is not None
            ):
                self._message_ingress_guard.release(
                    message_reservation_id,
                    message_reservation_owner,
                )

        message_id = ""
        try:
            ingress_facts = self._extract_canonical_ingress_facts(data)
            if ingress_facts is None:
                audit_logger.warning(
                    "INGRESS_MALFORMED_EVENT_REJECTED phase=worker"
                )
                return
            message_id, chat_id, chat_type, _sender_id = ingress_facts
            event = data.event
            message = event.message

            event_sender_union_id = getattr(
                getattr(getattr(event, "sender", None), "sender_id", None),
                "union_id",
                "",
            )
            _sender_union_id = (
                event_sender_union_id
                if isinstance(event_sender_union_id, str)
                else ""
            )
            _is_p2p = chat_type == "p2p"
            event_tenant_key = getattr(
                getattr(data, "header", None),
                "tenant_key",
                "",
            )
            _tenant_key = (
                event_tenant_key if isinstance(event_tenant_key, str) else ""
            )

            current_trust = self._resolve_effective_trust(
                sender_id=_sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
            )
            trust_decision = self._message_trust_access_decision(
                current_trust,
                chat_type=chat_type,
            )
            if trust_decision is not None and not trust_decision.allowed:
                return
            if effective_trust is not None and current_trust != effective_trust:
                audit_logger.warning(
                    "MANAGED_INGRESS_STALE_TRUST_DENIED chat_hash=%s",
                    self._access_identifier_hash(chat_id),
                )
                return

            # Parsing is allowed only after the current event facts pass their
            # independent worker check. No downstream fact is trusted from the
            # queued TaskSpec.
            image_handler = self._get_image_handler()
            parse_result = image_handler.parse_message(message.message_type, message.content)
            text = self._clean_at_text(parse_result.text)

            # Slash parsing is request-scoped: parse once and reuse.
            # This match becomes the single source of truth for downstream
            # slash consumers (gate/system/engines).
            try:
                command_match = SlashCommandParser.parse(text)
            except Exception:
                command_match = None

            ingress_decision = trust_decision or self._decide_ingress_access(
                message_id=message_id,
                sender_id=_sender_id,
                chat_id=chat_id,
                chat_type=chat_type,
                command_match=command_match,
            )
            if not ingress_decision.allowed:
                return
            request_id = self._handler_ctx.handlers["coco"].ensure_request_id(message_id, chat_id=chat_id)

            # Validation follows authorization so denied traffic cannot mutate
            # duplicate/expiry state or trigger unsupported-content replies.
            if not self._validate_message(
                message,
                request_id,
                message_reservation_owner=message_reservation_owner,
            ):
                return

            if (
                ingress_decision.operation not in _BOOTSTRAP_ACCESS_OPERATIONS
                and not self._managed_ingress_action_allowed(
                    current_trust,
                    text=text,
                    command_match=command_match,
                )
            ):
                return

            if ingress_decision.operation is AccessOperation.BOOTSTRAP_HELP:
                if command_match is None:
                    logger.critical(
                        "INGRESS_BOOTSTRAP_HELP_MATCH_MISSING"
                    )
                    return
                set_current_sender_id(_sender_id)
                set_current_sender_union_id(_sender_union_id or None)
                set_current_is_p2p(_is_p2p)
                set_current_tenant_key(_tenant_key or None)
                self._handler_ctx.handlers["system"].show_bootstrap_help(
                    message_id,
                    chat_id,
                )
                return

            if ingress_decision.operation in {
                AccessOperation.BOOTSTRAP_ADMIN,
                AccessOperation.ENROL_CURRENT_CHAT,
            }:
                if command_match is None:
                    logger.critical(
                        "INGRESS_ENROLMENT_COMMAND_MATCH_MISSING operation=%s",
                        ingress_decision.operation.value,
                    )
                    return
                # Enrollment is its own narrow route. It must not touch chat
                # locks, GroupLedger, image downloads, Shell, project lookup,
                # or any non-system handler.
                set_current_sender_id(_sender_id)
                set_current_sender_union_id(_sender_union_id or None)
                set_current_is_p2p(_is_p2p)
                set_current_tenant_key(_tenant_key or None)
                self._handler_ctx.handlers["system"].handle_intercepted_command(
                    message_id,
                    chat_id,
                    text,
                    command_match=command_match,
                )
                return

            # Propagate authorized identity to downstream request-local state.
            set_current_sender_id(_sender_id)
            set_current_sender_union_id(_sender_union_id or None)
            set_current_is_p2p(_is_p2p)
            set_current_tenant_key(_tenant_key or None)

            from .user_cache import resolve_display_name_nonblocking

            _display_name = (
                resolve_display_name_nonblocking(_sender_id, self._get_api_client)
                if _sender_id
                else ""
            )
            set_current_sender_name(
                _display_name or (_sender_id[:8] if _sender_id else "")
            )
            structured_mentions = tuple(
                dict.fromkeys(
                    name
                    for mention in (getattr(message, "mentions", None) or ())
                    if isinstance((name := getattr(mention, "name", None)), str)
                    and name
                    and name == name.strip()
                    and (
                        not getattr(mention, "tenant_key", None)
                        or not _tenant_key
                        or getattr(mention, "tenant_key", None) == _tenant_key
                    )
                )
            )
            set_current_mentioned_names(structured_mentions)

            root_id = getattr(message, "root_id", None)
            if (
                root_id
                and self.settings.thread_programming_enabled
                and not (
                    current_trust is not None
                    and current_trust.zone is TrustZone.MANAGED_AGENT_GROUP
                )
            ):
                thread_ctx = self._thread_manager.get(root_id)
                if thread_ctx:
                    set_current_thread_id(thread_ctx.thread_root_id)
                    logger.debug(
                        "[Thread] _process_async hit: msg_root=%s canonical=%s",
                        root_id[:12],
                        thread_ctx.thread_root_id[:12],
                    )

            # Chat lock interception (fail-close: non-admin blocked on exception).
            # Use the request-scoped CommandMatch instead of re-parsing raw text.
            if self._chat_lock_gate.check(
                chat_id,
                _sender_id,
                message_id,
                command_match=command_match,
            ):
                return

            # 3. Handle Images (if any)
            is_image_only = False
            trusted_project = None
            if (
                current_trust is not None
                and current_trust.managed_group is not None
            ):
                managed_group = current_trust.managed_group
                trusted_project = self._project_manager.get_project_for_chat(
                    managed_group.project_id,
                    chat_id,
                )
                if trusted_project is None:
                    return
                if (
                    getattr(trusted_project, "project_id", None)
                    != managed_group.project_id
                    or getattr(trusted_project, "root_path", None)
                    != managed_group.canonical_root_ref
                ):
                    return
            if parse_result.image_keys:
                image_kwargs = {}
                if trusted_project is not None:
                    image_kwargs["trusted_project"] = trusted_project
                project, auto_enter_mode, text, is_image_only = self._handle_image_content(
                    message,
                    parse_result.image_keys,
                    text,
                    request_id,
                    task_ctx,
                    **image_kwargs,
                )
                # Downloaded image references are part of the effective prompt.
                # Refresh slash args so engine consumers receive the
                # same evidence-rich goal as text-based engine handlers.
                if command_match is not None:
                    enriched_match = SlashCommandParser.parse(text)
                    if enriched_match is not None and enriched_match.command == command_match.command:
                        command_match = enriched_match
                if trusted_project is not None:
                    project = trusted_project
            else:
                # 4. Resolve Context (if no images to drive it)
                if trusted_project is not None:
                    project, auto_enter_mode = trusted_project, None
                else:
                    project, auto_enter_mode = self._resolve_message_context(message)

            # 4b. Safety net: if auto_enter_mode is still None but we are in a
            # registered thread, force-set mode from thread context so that the
            # message never accidentally falls through to SMART / intent recognition.
            if not auto_enter_mode and trusted_project is None:
                _root = getattr(message, "root_id", None)
                if _root and self.settings.thread_programming_enabled:
                    _tctx = self._thread_manager.get(_root)
                    if _tctx and _tctx.mode and _tctx.mode != "smart":
                        auto_enter_mode = _tctx.mode
                        set_current_thread_id(_tctx.thread_root_id)
                        thread_project = self._project_manager.get_project_for_chat(
                            _tctx.project_id,
                            chat_id,
                        )
                        if _tctx.mode in {"deep", "spec", "workflow"}:
                            project = thread_project
                        elif not project:
                            project = thread_project or self._project_manager.get_active_project(chat_id)
                        logger.info(
                            "[Thread] Safety-net resolved mode: root=%s canonical=%s mode=%s",
                            _root[:12], _tctx.thread_root_id[:12], auto_enter_mode,
                        )

            # 5. Handle Context Updates (Task Scheduler)
            if task_ctx and project:
                self._update_task_project(task_ctx, project.project_id)

            # 6. Dispatch Logic
            if not self._current_trust_can_dispatch(current_trust, project=project):
                return
            if not text and not is_image_only:
                # Special case: handle empty text (e.g. unsupported content that parsed to empty)
                # But wait, if image_keys exist, text might be empty but valid (image only).
                # _handle_image_content handles text augmentation.
                # If we are here and text is empty, check if we should show help or dispatch to mode.
                _root_id = getattr(message, "root_id", None)
                self._dispatch_empty_text(message_id, chat_id, project, task_ctx, root_id=_root_id)
                return

            self._dispatch_message_logic(
                message_id,
                chat_id,
                text,
                project,
                auto_enter_mode,
                command_match=command_match,
                is_image_only=is_image_only,
                shell_fast_tracked=shell_fast_tracked,
                chat_type=chat_type,
                effective_trust=current_trust,
            )

        except asyncio.TimeoutError as e:
            logger.warning("处理消息超时: %s", get_error_detail(e))
            try:
                self._handler_ctx.handlers["coco"].reply_text(message_id, UI_TEXT["ws_message_timeout"])
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError, AttributeError, LookupError):
                classify_ws_error(RuntimeError("reply timeout failed"), phase="dispatch")
                logger.debug("failed to reply timeout message", exc_info=True)
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError, AttributeError, LookupError) as e:
            classification = classify_ws_error(e, phase="dispatch")
            if classification.action == WSErrorAction.REPLY_INTERNAL_ERROR:
                logger.error("处理消息异常: %s", get_error_detail(e), exc_info=True)
                try:
                    self._handler_ctx.handlers["coco"].reply_text(message_id, UI_TEXT["ws_message_internal_error"])
                except (RuntimeError, OSError, TimeoutError, TypeError, ValueError, AttributeError, LookupError):
                    classify_ws_error(RuntimeError("reply internal error failed"), phase="best_effort_notify")
                    logger.debug("failed to reply internal error message", exc_info=True)
            elif classification.action == WSErrorAction.LOG_AND_CONTINUE:
                logger.debug("处理消息 best-effort 失败: %s", get_error_detail(e), exc_info=True)
            elif classification.action == WSErrorAction.PROPAGATE:
                logger.error("处理消息异常: %s", get_error_detail(e), exc_info=True)
                raise
        finally:
            set_current_thread_id(None)
            set_current_sender_id(None)
            set_current_sender_union_id(None)
            set_current_sender_name("")
            set_current_is_p2p(False)
            set_current_tenant_key(None)
            if message_id:
                with self._pending_image_lock:
                    self._pending_image_keys.pop(message_id, None)
                    self._pending_image_only.discard(message_id)
            if (
                message_reservation_id is not None
                and message_reservation_owner is not None
            ):
                self._message_ingress_guard.commit(
                    message_reservation_id,
                    message_reservation_owner,
                )

    def _validate_message(
        self,
        message,
        request_id: str,
        *,
        message_reservation_owner: str | None = None,
    ) -> bool:
        """校验消息是否需要处理（过期/重复/类型不支持等）。"""
        if message.create_time and self._message_ingress_guard.is_message_expired(
            int(message.create_time)
        ):
            logger.debug("跳过过期消息: %s", message.message_id)
            return False

        if message_reservation_owner is not None:
            if not self._message_ingress_guard.owns(
                message.message_id,
                message_reservation_owner,
            ):
                logger.debug("跳过无主消息占位: %s", message.message_id)
                return False
        elif self._message_ingress_guard.is_duplicate_message(message.message_id):
            logger.debug("跳过重复消息: %s", message.message_id)
            return False

        supported_types = {"text", "image", "post"}
        if message.message_type not in supported_types:
            self._handler_ctx.handlers["coco"].reply_text(message.message_id, UI_TEXT["ws_unsupported_msg_type"])
            return False
        return True

    def _clean_at_text(self, text: str) -> str:
        """移除 '@机器人' 前缀，得到用户真实输入文本。"""
        text = text.strip()
        if text.startswith("@"):
            parts = text.split(None, 1)
            if len(parts) > 1:
                return parts[1].strip()
            return ""
        return text

    def _handle_image_content(
        self,
        message,
        image_keys,
        text,
        request_id,
        task_ctx,
        *,
        trusted_project=None,
    ):
        """处理图片消息：下载并把图片引用文本拼接回 prompt。

        返回 `(project, auto_enter_mode, text, is_image_only)`。
        """
        message_id = message.message_id
        chat_id = message.chat_id
        getattr(message, "parent_id", None)
        getattr(message, "root_id", None)

        with self._pending_image_lock:
            self._pending_image_keys[message_id] = image_keys

        if trusted_project is not None:
            project, auto_enter_mode = trusted_project, None
        else:
            project, auto_enter_mode = self._resolve_message_context(message)

        try:
            if project:
                self._message_linker.register_origin(
                    message_id, request_id=request_id, chat_id=chat_id, project_id=project.project_id
                )
        except Exception as e:
            logger.debug("register_origin失败(image_msg): message_id=%s, err=%s", message_id, get_error_detail(e))

        if task_ctx and project:
            self._update_task_project(task_ctx, project.project_id)

        save_dir = FeishuImageHandler.get_image_save_dir()

        image_handler = self._get_image_handler()
        download_result = image_handler.download_images(message_id, image_keys, save_dir)

        is_image_only = False
        if download_result.saved_paths:
            is_image_only = not text
            ref_text = FeishuImageHandler.build_image_reference_text(download_result.saved_paths)
            if text:
                text += ref_text
            else:
                text = UI_TEXT["ws_image_only_prefix"] + ref_text

        if download_result.failed_keys:
            logger.warning("部分图片下载失败: %s", download_result.failed_keys)

        if is_image_only:
            with self._pending_image_lock:
                self._pending_image_only.add(message_id)

        return project, auto_enter_mode, text, is_image_only

    def _resolve_message_context(self, message):
        """从 message 的 parent/root 引用恢复项目上下文。

        优先级：话题上下文 > 消息引用上下文 > 当前 active project。
        """
        message_id = message.message_id
        chat_id = message.chat_id
        parent_id = getattr(message, "parent_id", None)
        root_id = getattr(message, "root_id", None)

        if root_id and self.settings.thread_programming_enabled:
            thread_ctx = self._thread_manager.get(root_id)
            if thread_ctx:
                from ..thread import set_current_thread_id
                set_current_thread_id(thread_ctx.thread_root_id)
                auto_enter_mode = thread_ctx.mode if thread_ctx.mode != "smart" else None
                project = self._project_manager.get_project_for_chat(thread_ctx.project_id, chat_id)
                if not project and thread_ctx.mode not in {"deep", "spec", "workflow"}:
                    project = self._project_manager.get_active_project(chat_id)
                logger.info(
                    "[Thread] Resolved context: msg_root=%s canonical=%s project=%s mode=%s project_found=%s",
                    root_id[:12], thread_ctx.thread_root_id[:12], thread_ctx.project_id, thread_ctx.mode, project is not None,
                )
                return project, auto_enter_mode

        return self._resolve_project_from_message(message_id, chat_id, parent_id or root_id)

    def _get_effective_mode(self, chat_id: str, project_id: Optional[str] = None):
        """Return the effective interaction mode and whether it is a programming mode."""
        from ..mode import InteractionMode
        thread_id = get_current_thread_id()
        if thread_id:
            thread_ctx = self._thread_manager.get(thread_id)
            if thread_ctx and thread_ctx.mode != "smart":
                try:
                    return InteractionMode(thread_ctx.mode), True
                except ValueError:
                    logger.debug(
                        "thread mode is engine-only, not InteractionMode: %s",
                        thread_ctx.mode,
                    )
                    return InteractionMode.SMART, False
        return (
            self._mode_manager.get_mode(chat_id, project_id=project_id),
            self._mode_manager.is_programming_mode(chat_id, project_id=project_id),
        )

    def _is_topic_engine_context(self) -> bool:
        """Return True when the current Feishu topic is owned by an engine."""
        thread_id = get_current_thread_id()
        if not thread_id or not self.settings.thread_programming_enabled:
            return False
        thread_ctx = self._thread_manager.get(thread_id)
        return bool(thread_ctx and thread_ctx.mode in {"deep", "spec", "workflow"})

    def _get_mode_handler(self, mode):
        """Look up the registered handler corresponding to an interaction mode."""
        return self._handler_ctx.handlers.get(getattr(mode, "value", ""))

    def _find_active_thread(self, chat_id):
        """Return the first active non-smart topic context for the given chat."""
        if not self.settings.thread_programming_enabled:
            return None
        contexts = self._thread_manager.get_by_chat(chat_id)
        for ctx in contexts:
            if ctx.mode and ctx.mode != "smart":
                return ctx
        return None

    def _update_task_project(self, task_ctx, project_id):
        """将调度任务与 project_id 关联（便于任务看板/诊断）。"""
        try:
            self._scheduler.update_project_id(task_ctx.run_id, project_id)
        except Exception as e:
            logger.debug("update_project_id失败: run_id=%s, err=%s", task_ctx.run_id, get_error_detail(e))

    def _dispatch_empty_text(self, message_id, chat_id, project, task_ctx, *, root_id=None):
        """处理"文本为空"的情况：在编程模式下仍转发（保持会话），否则展示帮助。"""

        _pid = project.project_id if project else None
        if not _pid and task_ctx and task_ctx.spec.project_id:
            _pid = task_ctx.spec.project_id

        current_mode, _is_prog = self._get_effective_mode(chat_id, project_id=_pid)
        if current_mode in PROGRAMMING_MODES:
            if project is None:
                project = self._project_manager.get_active_project(chat_id)

            handler = self._get_mode_handler(current_mode)
            if handler:
                handler.handle_message(message_id, chat_id, "", project)
        else:
            self._handler_ctx.handlers["system"].show_help(message_id, chat_id)

    def _dispatch_message_logic(
        self,
        message_id,
        chat_id,
        text,
        project,
        auto_enter_mode,
        *,
        command_match,
        is_image_only=False,
        shell_fast_tracked=False,
        chat_type: str = "group",
        effective_trust: EffectiveTrust | None = None,
    ):
        """Apply topic/project safety gates, then use the one dispatcher."""
        handlers = self._handler_ctx.handlers

        def dispatch(target_project=project) -> None:
            if not self._current_trust_can_dispatch(
                effective_trust, project=target_project
            ):
                return
            self._message_dispatcher.process_with_intent(
                message_id,
                chat_id,
                text,
                target_project,
                command_match=command_match,
                shell_fast_tracked=shell_fast_tracked,
                chat_type=chat_type,
                effective_trust=effective_trust,
            )

        engine_modes = {"deep", "spec", "workflow"}
        if auto_enter_mode in engine_modes:
            command = getattr(command_match, "command", "")
            args = str(getattr(command_match, "args", "")).lower()
            is_exit = SystemHandler.is_exit_command(text)
            safe_without_project = command in {
                "/help", "/projects", "/project", "/status", "/exit", "/quit",
            } or (
                command in {"/stop_deep", "/deep_status"}
                and args in {"all", "-a", "--all"}
            )
            if project is None:
                if is_exit and self._current_trust_can_dispatch(effective_trust):
                    handlers["system"].exit_current_mode(
                        message_id, chat_id, project=None
                    )
                elif safe_without_project:
                    dispatch()
                else:
                    handlers["coco"].reply_text(
                        message_id, UI_TEXT["ws_topic_project_unavailable"]
                    )
                return

            requested = None
            if command in {"/deep", "/deep_update", "/deep_status", "/stop_deep"}:
                requested = "deep"
            elif command.startswith("/spec") or command == "/stop_spec":
                requested = "spec"
            else:
                from ..workflow_engine.commands import TOPIC_ENGINE_COMMANDS

                if command in TOPIC_ENGINE_COMMANDS:
                    requested = "workflow"
            if requested and requested != auto_enter_mode:
                handlers["coco"].reply_text(
                    message_id,
                    UI_TEXT["topic_engine_switch_blocked"].format(
                        current=auto_enter_mode.upper(),
                        requested=requested.upper(),
                    ),
                )
                return
            if command_match is None:
                if not self._current_trust_can_dispatch(
                    effective_trust, project=project
                ):
                    return
                handlers["coco"].add_reaction(
                    message_id, EmojiReaction.on_processing()
                )
                if auto_enter_mode == "workflow":
                    handlers["workflow"].handle_message(
                        message_id, chat_id, text, project
                    )
                else:
                    getattr(
                        handlers[auto_enter_mode],
                        f"start_{auto_enter_mode}_engine",
                    )(message_id, chat_id, text, project)
                return

        managed = (
            effective_trust is not None
            and effective_trust.zone is TrustZone.MANAGED_AGENT_GROUP
        )
        if (
            auto_enter_mode is None
            and not managed
            and command_match is None
            and not is_image_only
            and text
            and not self._intent_recognizer.looks_like_shell(text)
        ):
            bound_project = self._project_manager.find_by_bound_chat_id(chat_id)
            if bound_project is not None:
                project_id = getattr(bound_project, "project_id", None)
                _, is_programming = self._get_effective_mode(
                    chat_id, project_id=project_id
                )
                if is_programming:
                    dispatch(bound_project)
                    return
                tool = str(
                    get_project_acp_tool_name(bound_project, chat_id)
                    or getattr(self.settings, "default_acp_tool", None)
                    or "coco"
                ).strip().lower()
                handler = handlers[
                    tool if tool in PROGRAMMING_MODE_VALUES else "coco"
                ]
                if not self._current_trust_can_dispatch(
                    effective_trust, project=bound_project
                ):
                    return
                handlers["coco"].add_reaction(
                    message_id, EmojiReaction.on_coco_mode()
                )
                handlers["coco"].add_reaction(
                    message_id, EmojiReaction.on_processing()
                )
                handler.enter_mode(
                    message_id, chat_id, silent=True, project=bound_project
                )
                handler.handle_message(message_id, chat_id, text, bound_project)
                return
        dispatch()
