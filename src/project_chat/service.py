"""ProjectChatService — orchestrator for /new-chat command."""

import logging
import os
import subprocess
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..config import get_settings
from ..project.context import ProjectContext
from ..project.manager import ProjectManager
from .errors import CreateChatError
from .group_naming import format_group_name, validate_name_part
from .lark_chat_client import LarkChatClient

if TYPE_CHECKING:
    from ..trust.registry import ManagedGroupRegistry

logger = logging.getLogger(__name__)

# Per-root lock: one provision ID/root may be entered from multiple source chats.
_creation_locks: dict[str, threading.Lock] = {}
_creation_locks_guard = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _get_creation_lock(chat_id: str, path: str) -> threading.Lock:
    del chat_id
    key = os.path.normcase(os.path.realpath(os.path.normpath(path)))
    with _creation_locks_guard:
        if key not in _creation_locks:
            _creation_locks[key] = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        return _creation_locks[key]


class ProjectChatService:
    """Orchestrates /new-chat: parse → idempotency check → create chat → bind project."""

    def __init__(
        self,
        project_manager: ProjectManager,
        lark_chat_client: LarkChatClient,
        reply_fn: Callable[[str, str, Optional[str]], Any],
        send_to_chat_fn: Callable[[str, str, str, Optional[str]], Any],
        managed_group_registry: Optional["ManagedGroupRegistry"] = None,
        owner_id: str = "",
        receiving_bot_ref: str = "",
    ):
        self._pm = project_manager
        self._lark = lark_chat_client
        self._reply = reply_fn
        self._send_to_chat = send_to_chat_fn
        self._managed_groups = managed_group_registry
        self._owner_id = owner_id
        self._receiving_bot_ref = receiving_bot_ref
        if managed_group_registry is not None and (
            not owner_id or not receiving_bot_ref
        ):
            raise ValueError(
                "owner_id and receiving_bot_ref are required with managed_group_registry"
            )

    def handle(
        self,
        message_id: str,
        chat_id: str,
        sender_open_id: str,
        data: dict,
    ) -> None:
        """Main entry point for /new-chat command."""
        settings = get_settings()

        # 1. Parse defaults
        path = data.get("path") or os.getcwd()
        path = os.path.expanduser(os.path.abspath(path))
        name = data.get("name") or os.path.basename(os.path.normpath(path)) or f"project_{int(time.time())}"
        suffix = data.get("suffix") or settings.project_chat_suffix

        # Validate name/suffix
        err = validate_name_part(name)
        if err:
            self._reply(message_id, f"❌ 项目名无效: {err}", None)
            return
        err = validate_name_part(suffix)
        if err:
            self._reply(message_id, f"❌ 后缀无效: {err}", None)
            return

        # 2. Acquire per-(chat, path) lock
        lock = _get_creation_lock(chat_id, path)
        if not lock.acquire(timeout=5):
            self._reply(message_id, "⏳ 正在处理中，请稍后再试", None)
            return

        try:
            self._handle_locked(message_id, chat_id, sender_open_id, name, suffix, path)
        finally:
            lock.release()

    def _handle_locked(
        self,
        message_id: str,
        chat_id: str,
        sender_open_id: str,
        name: str,
        suffix: str,
        path: str,
    ) -> None:
        from ..trust.models import ManagedGroupOrigin

        provision_id = (
            f"new-chat:{self._owner_id}:{ProjectManager.generate_id(name)}:"
            f"{os.path.normpath(path)}"
        )
        # 3. Idempotency check — chat_id=None to skip visibility filter
        ctx = self._pm.find_project_by_path(path, chat_id=None)

        # 3.1 Fallback: find by name (legacy project may have mismatched root_path)
        if ctx is None:
            ctx = self._pm.find_project_by_name(name, chat_id=None)
            if ctx:
                if self._pm.pending_managed_chat_binding_sagas_for_project(
                    ctx.project_id
                ):
                    self._reply(
                        message_id,
                        "⚠️ 项目存在未完成的受管群绑定事务，禁止修改根目录；请先重试原绑定。",
                        None,
                    )
                    return
                # Update root_path/working_dir to current path for legacy project
                ctx.root_path = path
                ctx.working_dir = path
                if self._pm._save_projects() is False:
                    self._reply(message_id, "❌ 更新历史项目失败，请重试", None)
                    return
                logger.info(
                    "Legacy project %s found by name, updated root_path to %s",
                    ctx.project_id, path,
                )

        if ctx and ctx.bound_chat_id:
            if self._managed_groups is not None:
                active = self._managed_groups.active_record(ctx.bound_chat_id)
                if (
                    active is None
                    or active.project_id != ctx.project_id
                    or active.canonical_root_ref != ctx.root_path
                    or active.origin is not ManagedGroupOrigin.GHOSTAP_CREATED
                    or active.owner_id != self._owner_id
                    or active.receiving_bot_ref != self._receiving_bot_ref
                ):
                    self._reply(
                        message_id,
                        "❌ 现有项目群尚未通过受管群校验，请由 Owner 在私聊中导入或重新绑定。",
                        None,
                    )
                    return
                grant = self._managed_groups.grant_for_chat(ctx.bound_chat_id)
                if (
                    grant is None
                    or grant.project_id != ctx.project_id
                    or grant.canonical_root_ref != ctx.root_path
                    or grant.managed_group_id != ctx.bound_chat_id
                    or grant.owner_id != self._owner_id
                    or grant.backend_binding_ids
                    or grant.connected_target_refs
                ):
                    self._reply(
                        message_id,
                        "❌ 现有项目群授权事实不完整，已保持失败关闭。",
                        None,
                    )
                    return
                pending = self._pm.pending_managed_chat_binding_sagas_for_project(
                    ctx.project_id,
                    ctx.bound_chat_id,
                )
                if pending:
                    saga = self._pm.resolve_managed_chat_binding_saga(
                        project_id=ctx.project_id,
                        chat_id=ctx.bound_chat_id,
                        expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                        expected_owner_id=self._owner_id,
                        expected_receiving_bot_ref=self._receiving_bot_ref,
                        expected_root_ref=ctx.root_path,
                    )
                    if saga is None or (
                        not self._pm.validate_managed_chat_binding_saga(
                            saga.operation_id
                        )
                        or not self._pm.complete_managed_chat_binding_saga(
                            saga.operation_id
                        )
                    ):
                        self._reply(
                            message_id,
                            "⚠️ 项目群已登记，但本地绑定事务尚未安全收尾，请重试。",
                            None,
                        )
                        return
            # Branch A: already bound → ensure originating chat can see it, then return jump card
            if chat_id and chat_id not in ctx.allowed_chat_ids:
                ctx.add_chat_id(chat_id)
                if self._pm._save_projects() is False:
                    self._reply(message_id, "❌ 更新项目可见范围失败，请重试", None)
                    return
            self._reply_jump_card(message_id, ctx)
            return

        group_name = format_group_name(name, suffix)
        description = self._build_description(name, path)
        intended_project_id = ctx.project_id if ctx else ProjectManager.generate_id(name)
        recovered_chat_id: str | None = None

        if self._managed_groups is not None:
            from ..trust.models import ManagedGroupOrigin

            try:
                self._managed_groups.begin_provision(
                    provision_id=provision_id,
                    owner_id=self._owner_id,
                    origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                    receiving_bot_ref=self._receiving_bot_ref,
                    project_id=intended_project_id,
                    canonical_root_ref=path,
                    created_at=datetime.now(UTC),
                )
                recovered_chat_id = self._managed_groups.provision_chat_id(
                    provision_id
                )
            except Exception:
                logger.exception(
                    "managed group provision intent failed for project=%s",
                    intended_project_id,
                )
                self._reply(message_id, "❌ 受管群登记准备失败，请重试", None)
                return

        residual = self._pm.managed_group_residual(provision_id)
        if residual is not None and residual[1] not in {
            "create_outcome_unknown",
            "registry_bind_uncertain",
        }:
            self._reply(
                message_id,
                "⚠️ 上次建群留下未确认的飞书群，已阻止重复建群；请先人工处理残留群。",
                None,
            )
            return

        # 4. Create the remote chat once, then durably bind its ID before any
        # local project mutation.  A retry after restart reuses that binding.
        if recovered_chat_id is not None:
            from .lark_chat_client import ManagedChatValidation

            validation = self._lark.validate_managed_chat(
                recovered_chat_id,
                self._owner_id,
            )
            if validation is not ManagedChatValidation.VALID:
                self._pm.record_managed_group_residual(
                    provision_id,
                    recovered_chat_id,
                    "recovered_chat_invalid",
                )
                self._reply(
                    message_id,
                    "⚠️ 无法确认重试群仍存在且 Owner/接收 Bot 均在群内，已停止激活。",
                    None,
                )
                return
            new_chat_id = recovered_chat_id
            new_chat_name = group_name
        else:
            if self._managed_groups is not None and not (
                self._managed_groups.prepare_create_dispatch(
                    provision_id,
                    dispatched_at=datetime.now(UTC),
                )
            ):
                self._pm.record_managed_group_residual(
                    provision_id,
                    "outcome_unknown",
                    "create_outcome_unknown",
                )
                self._reply(
                    message_id,
                    "⚠️ 上次建群结果仍不确定且去重窗口已过，已阻止重复建群；"
                    "请人工核对飞书群。",
                    None,
                )
                return
            try:
                result = self._lark.create_chat(
                    name=group_name,
                    description=description,
                    user_id_list=[sender_open_id],
                    operation_id=provision_id,
                )
            except CreateChatError as e:
                from .errors import CreateChatFailureDisposition

                logger.warning("create_chat failed for path=%s: %s", path, str(e))
                self._reply(message_id, f"❌ 建群失败: {e}", None)
                if (
                    self._managed_groups is not None
                    and e.disposition
                    is CreateChatFailureDisposition.OUTCOME_UNKNOWN
                ):
                    self._managed_groups.mark_create_outcome_unknown(provision_id)
                    self._pm.record_managed_group_residual(
                        provision_id,
                        "outcome_unknown",
                        "create_outcome_unknown",
                    )
                else:
                    self._abandon_provision(provision_id)
                return
            new_chat_id = result.chat_id
            new_chat_name = result.name
            if self._managed_groups is not None:
                try:
                    self._managed_groups.bind_provision_chat(
                        provision_id, new_chat_id
                    )
                except Exception as exc:
                    from ..trust.registry import RegistryCommitUncertainError

                    logger.exception(
                        "failed to bind remote chat to provision %s", provision_id
                    )
                    if (
                        isinstance(exc, RegistryCommitUncertainError)
                        and exc.committed
                    ):
                        self._pm.record_managed_group_residual(
                            provision_id,
                            new_chat_id,
                            "registry_bind_uncertain",
                        )
                        self._reply(
                            message_id,
                            "⚠️ 受管群远端绑定结果不确定，已保留飞书群且不会执行删除；"
                            "服务恢复核对后可安全重试。",
                            None,
                        )
                        return
                    self._record_untrusted_remote(provision_id, new_chat_id)
                    self._reply(
                        message_id,
                        "⚠️ 受管群远端绑定持久化失败；为避免竞态删错群，"
                        "已保留该群但不授予信任，请重试验证或由 Owner 处理。",
                        None,
                    )
                    return

        # 4.5 Promote sender to group manager (best-effort, enables dissolve permission)
        self._lark.add_managers(new_chat_id, [sender_open_id])

        # 5. Bind, then atomically activate Registry before any success delivery.
        created_project = False
        binding_saga_prepared = False
        try:
            if ctx is None:
                success, msg, ctx_new = self._pm.create_project_with_managed_chat_saga(
                    project_id=ProjectManager.generate_id(name),
                    project_name=name,
                    root_path=path,
                    owner_chat_id=new_chat_id,
                    additional_chat_id=chat_id,
                    managed_chat_id=new_chat_id,
                    managed_chat_name=new_chat_name,
                    created_at=time.time(),
                    operation_id=provision_id,
                    expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                    expected_owner_id=self._owner_id,
                    expected_receiving_bot_ref=self._receiving_bot_ref,
                )
                if not success or not ctx_new:
                    self._record_untrusted_remote(provision_id, new_chat_id)
                    self._reply(
                        message_id,
                        f"⚠️ 创建项目失败: {msg}；远端群已保留但不授予信任，请由 Owner 核对。",
                        None,
                    )
                    return
                created_project = True
                ctx = ctx_new
                binding_saga_prepared = True
            else:
                bound, _binding_snapshot = self._pm.bind_managed_chat_for_saga(
                    ctx.project_id,
                    new_chat_id,
                    chat_name=new_chat_name,
                    created_at=time.time(),
                    operation_id=provision_id,
                    additional_chat_id=chat_id,
                    project_name=name,
                    remove_project_on_restore=False,
                    expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                    expected_owner_id=self._owner_id,
                    expected_receiving_bot_ref=self._receiving_bot_ref,
                )
                if not bound:
                    raise OSError("project binding persistence failed")
                binding_saga_prepared = True

            if self._managed_groups is not None:
                self._managed_groups.activate(
                    provision_id=provision_id,
                    chat_id=new_chat_id,
                    project_id=ctx.project_id,
                    canonical_root_ref=ctx.root_path,
                )
            if not self._pm.complete_managed_chat_binding_saga(provision_id):
                self._reply(
                    message_id,
                    "⚠️ 项目群已登记，但本地绑定事务收尾失败；请重试以完成恢复核对。",
                    None,
                )
                return
        except Exception as e:
            from ..project.manager import ProjectCommitUncertainError
            from ..trust.registry import RegistryCommitUncertainError

            if isinstance(
                e, (RegistryCommitUncertainError, ProjectCommitUncertainError)
            ) and e.committed:
                logger.error(
                    "registry activation durability is uncertain; preserving "
                    "remote chat and durable project binding chat=%s",
                    new_chat_id[:12],
                )
                self._reply(
                    message_id,
                    "⚠️ 项目群登记结果不确定，已停止继续操作且不会删除飞书群；"
                    "服务恢复核对前该群不获得信任。",
                    None,
                )
                return
            logger.error(
                "bind/registry activation failed, rolling back chat %s: %s",
                new_chat_id[:12],
                str(e),
            )
            rollback_ok = (
                self._pm.restore_managed_chat_binding_saga(provision_id)
                if binding_saga_prepared
                else self._rollback_binding(
                    ctx=ctx,
                    created_project=created_project,
                    existing_snapshot=None,
                    remote_chat_id=new_chat_id,
                )
            )
            self._record_untrusted_remote(provision_id, new_chat_id)
            detail = "远端群已保留但未获得信任，请由 Owner 重试验证或处理"
            if not rollback_ok:
                detail += "；本地补偿持久化失败，绑定已隔离，请人工检查"
            self._reply(message_id, f"❌ 项目群绑定失败，{detail}", None)
            return

        # 6. Reply in main chat + welcome in new chat
        self._reply_jump_card(message_id, ctx)
        self._send_welcome(new_chat_id, ctx)

    def _abandon_provision(self, provision_id: str) -> None:
        if self._managed_groups is None:
            return
        try:
            self._managed_groups.abandon_provision(provision_id)
        except Exception:
            logger.exception("failed to abandon managed group provision")

    def _record_untrusted_remote(
        self,
        provision_id: str,
        chat_id: str,
    ) -> None:
        self._pm.record_managed_group_residual(
            provision_id,
            chat_id,
            "untrusted_retained",
        )

    @staticmethod
    def _binding_snapshot(ctx: ProjectContext) -> dict[str, Any]:
        return {
            "allowed_chat_ids": OrderedDict(ctx.allowed_chat_ids),
            "bound_chat_created_at": ctx.bound_chat_created_at,
            "bound_chat_id": ctx.bound_chat_id,
            "bound_chat_name": ctx.bound_chat_name,
            "owner_chat_id": ctx.owner_chat_id,
            "project_name": ctx.project_name,
        }

    def _rollback_binding(
        self,
        *,
        ctx: ProjectContext | None,
        created_project: bool,
        existing_snapshot: dict[str, Any] | None,
        remote_chat_id: str,
    ) -> bool:
        if ctx is None:
            return True
        if created_project:
            closed, _ = self._pm.close_project(ctx.project_id)
            if not closed:
                self._pm.quarantine_bound_chat(remote_chat_id)
            return closed
        if existing_snapshot is None:
            return True
        ctx.allowed_chat_ids = OrderedDict(existing_snapshot["allowed_chat_ids"])
        ctx.bound_chat_created_at = existing_snapshot["bound_chat_created_at"]
        ctx.bound_chat_id = existing_snapshot["bound_chat_id"]
        ctx.bound_chat_name = existing_snapshot["bound_chat_name"]
        ctx.owner_chat_id = existing_snapshot["owner_chat_id"]
        ctx.project_name = existing_snapshot["project_name"]
        saved = self._pm._save_projects()
        if not saved:
            self._pm.quarantine_bound_chat(remote_chat_id)
        return saved

    def _build_description(self, name: str, path: str) -> str:
        git_remote = self._detect_git_remote(path)
        lines = [
            f"🎯 项目: {name}",
            f"📁 目录: {path}",
        ]
        if git_remote:
            lines.append(f"🔗 仓库: {git_remote}")
        lines.append("🤖 在这个群直接对话即可：默认 Coco / 显式 /claude /codex 等。")
        return "\n".join(lines)

    @staticmethod
    def _detect_git_remote(path: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", path, "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=3,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _reply_jump_card(self, message_id: str, ctx: ProjectContext) -> None:
        """Reply with a jump card pointing to the bound chat."""
        from ..card.builders.project import ProjectBuilder

        msg_type, card_json = ProjectBuilder.build_project_chat_jump_card(ctx)
        self._reply(message_id, card_json, msg_type)

    def _send_welcome(self, chat_id: str, ctx: ProjectContext) -> None:
        """Send welcome message in the newly created group."""
        text = (
            f"🎉 项目 **{ctx.project_name}** 专属群已就绪\n"
            f"📂 目录: `{ctx.root_path}`\n\n"
            f"直接在这里对话即可开始编程：\n"
            f"• 直接发消息 → 默认 Coco\n"
            f"• `/claude` → Claude 模式\n"
            f"• `/codex` → Codex 模式\n"
            f"• `/deep <需求>` → Deep 深度执行"
        )
        try:
            self._send_to_chat(chat_id, "text", text, None)
        except Exception as e:
            logger.warning("send_welcome to %s failed: %s", chat_id[:12], str(e))
