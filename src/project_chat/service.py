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

# Per-(chat_id, path) lock to prevent concurrent /new-chat races
_creation_locks: dict[str, threading.Lock] = {}
_creation_locks_guard = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _get_creation_lock(chat_id: str, path: str) -> threading.Lock:
    key = f"{chat_id}:{os.path.normpath(path)}"
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
        # 3. Idempotency check — chat_id=None to skip visibility filter
        ctx = self._pm.find_project_by_path(path, chat_id=None)

        # 3.1 Fallback: find by name (legacy project may have mismatched root_path)
        if ctx is None:
            ctx = self._pm.find_project_by_name(name, chat_id=None)
            if ctx:
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
                ):
                    self._reply(
                        message_id,
                        "❌ 现有项目群尚未通过受管群校验，请由 Owner 在私聊中导入或重新绑定。",
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
        provision_id = (
            f"new-chat:{self._owner_id}:{ProjectManager.generate_id(name)}:"
            f"{os.path.normpath(path)}"
        )
        intended_project_id = ctx.project_id if ctx else ProjectManager.generate_id(name)

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
            except Exception:
                logger.exception(
                    "managed group provision intent failed for project=%s",
                    intended_project_id,
                )
                self._reply(message_id, "❌ 受管群登记准备失败，请重试", None)
                return

        # 4. Create chat
        try:
            result = self._lark.create_chat(
                name=group_name,
                description=description,
                user_id_list=[sender_open_id],
            )
        except CreateChatError as e:
            logger.warning("create_chat failed for path=%s: %s", path, str(e))
            self._reply(message_id, f"❌ 建群失败: {e}", None)
            self._abandon_provision(provision_id)
            return

        new_chat_id = result.chat_id
        new_chat_name = result.name

        # 4.5 Promote sender to group manager (best-effort, enables dissolve permission)
        self._lark.add_managers(new_chat_id, [sender_open_id])

        # 5. Bind, then atomically activate Registry before any success delivery.
        created_project = False
        existing_snapshot = self._binding_snapshot(ctx) if ctx else None
        try:
            if ctx:
                # Branch B: legacy project without bound chat
                ctx.project_name = name  # respect user-specified name
                ctx.bound_chat_id = new_chat_id
                ctx.bound_chat_name = new_chat_name
                ctx.bound_chat_created_at = time.time()
                ctx.add_chat_id(new_chat_id)
                # Ensure the originating chat can still see this project
                if chat_id != new_chat_id:
                    ctx.add_chat_id(chat_id)
                if self._pm._save_projects() is False:
                    raise OSError("project binding persistence failed")
            else:
                # Branch C: new project
                success, msg, ctx_new = self._pm.create_project(
                    project_id=None,
                    project_name=name,
                    root_path=path,
                    chat_id=new_chat_id,
                )
                if not success or not ctx_new:
                    # Rollback: delete the created chat
                    self._lark.delete_chat(new_chat_id)
                    self._reply(message_id, f"❌ 创建项目失败: {msg}", None)
                    self._abandon_provision(provision_id)
                    return
                created_project = True
                ctx_new.bound_chat_id = new_chat_id
                ctx_new.bound_chat_name = new_chat_name
                ctx_new.bound_chat_created_at = time.time()
                # Ensure the originating chat can also see this project
                if chat_id != new_chat_id:
                    ctx_new.add_chat_id(chat_id)
                if self._pm._save_projects() is False:
                    raise OSError("project binding persistence failed")
                ctx = ctx_new

            if self._managed_groups is not None:
                self._managed_groups.activate(
                    provision_id=provision_id,
                    chat_id=new_chat_id,
                    project_id=ctx.project_id,
                    canonical_root_ref=ctx.root_path,
                )
        except Exception as e:
            logger.error(
                "bind/registry activation failed, rolling back chat %s: %s",
                new_chat_id[:12],
                str(e),
            )
            self._rollback_binding(
                ctx=ctx,
                created_project=created_project,
                existing_snapshot=existing_snapshot,
            )
            self._abandon_provision(provision_id)
            delete_result = self._lark.delete_chat(new_chat_id)
            if delete_result is True:
                detail = "飞书群已删除"
            elif delete_result is False:
                detail = "飞书群删除失败，请手动删除；该群未获得信任"
            else:
                detail = "飞书删群结果未知，请人工确认；该群未获得信任"
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
    ) -> None:
        if ctx is None:
            return
        if created_project:
            self._pm.close_project(ctx.project_id)
            return
        if existing_snapshot is None:
            return
        ctx.allowed_chat_ids = OrderedDict(existing_snapshot["allowed_chat_ids"])
        ctx.bound_chat_created_at = existing_snapshot["bound_chat_created_at"]
        ctx.bound_chat_id = existing_snapshot["bound_chat_id"]
        ctx.bound_chat_name = existing_snapshot["bound_chat_name"]
        ctx.owner_chat_id = existing_snapshot["owner_chat_id"]
        ctx.project_name = existing_snapshot["project_name"]
        self._pm._save_projects()

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
