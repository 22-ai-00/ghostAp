"""Create and atomically bind a dedicated Feishu project group."""

import logging
import os
import re
import subprocess
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Callable, Optional

from ..config import get_settings
from ..project.context import ProjectContext
from ..project.manager import ProjectManager
from ..trust.models import ManagedGroupOrigin
from .lark_chat_client import (
    CreateChatError,
    CreateChatFailureDisposition,
    LarkChatClient,
    ManagedChatValidation,
)

if TYPE_CHECKING:
    from ..trust.registry import ManagedGroupRegistry

logger = logging.getLogger(__name__)
_creation_locks: dict[str, threading.Lock] = {}
_creation_locks_guard = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_VALID_NAME = re.compile(r"^[\w\-.]+$", re.UNICODE)


class _Abort(RuntimeError):
    """A fail-closed provision outcome suitable for the command reply."""


def _creation_lock(path: str) -> threading.Lock:
    key = os.path.normcase(os.path.realpath(os.path.normpath(path)))
    with _creation_locks_guard:
        return _creation_locks.setdefault(
            key,
            threading.Lock(),  # leaf lock: never held while acquiring a LockLevel lock
        )


def _name_error(part: str) -> str | None:
    part = part.strip()
    if not part:
        return "名称不能为空"
    if len(part) > 50:
        return "名称过长（最大 50 字符）"
    if not _VALID_NAME.match(part):
        return "名称包含非法字符（不能包含空格或特殊符号，允许字母/数字/中文/下划线/短横/点）"
    return None


class _Provision:
    """One registry/remote/local compensation boundary for a group provision."""

    def __init__(
        self,
        project_manager: ProjectManager,
        registry: "ManagedGroupRegistry",
        remote: LarkChatClient,
        provision_id: str,
        owner_id: str,
    ) -> None:
        self.pm = project_manager
        self.registry = registry
        self.remote = remote
        self.id = provision_id
        self.owner_id = owner_id

    def begin(self, project_id: str, root: str, bot_ref: str) -> str | None:
        try:
            self.registry.begin_provision(
                provision_id=self.id,
                owner_id=self.owner_id,
                origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                receiving_bot_ref=bot_ref,
                project_id=project_id,
                canonical_root_ref=root,
                created_at=datetime.now(UTC),
            )
            return self.registry.provision_chat_id(self.id)
        except Exception as exc:
            logger.exception("managed group provision intent failed")
            raise _Abort("❌ 受管群登记准备失败，请重试") from exc

    def record(self, chat_id: str, state: str) -> None:
        from ..project.manager import ProjectCommitUncertainError

        try:
            saved = self.pm.record_managed_group_residual(self.id, chat_id, state)
        except ProjectCommitUncertainError as exc:
            raise _Abort(
                "⚠️ 远端群已保留且未授信，但本地恢复记录耐久性不确定；已失败关闭"
            ) from exc
        if not saved:
            raise _Abort(
                "⚠️ 远端群已保留且未授信，但本地恢复记录持久化失败；当前进程已失败关闭"
            )

    def consume(self, residual: tuple[str, str]) -> None:
        from ..project.manager import ProjectCommitUncertainError

        try:
            consumed = self.pm.consume_managed_group_residual(
                self.id, residual[0], residual[1]
            )
        except ProjectCommitUncertainError as exc:
            raise _Abort("⚠️ 本地恢复记录清理耐久性不确定；请重试核对") from exc
        if not consumed:
            raise _Abort("⚠️ 本地恢复记录未能持久清理；请重试核对")

    def abandon(self) -> bool:
        try:
            return bool(self.registry.abandon_provision(self.id))
        except Exception:
            logger.exception("failed to abandon managed group provision")
            return False

    def retain(self, chat_id: str) -> None:
        self.record(chat_id, "untrusted_retained")

    def _bind_remote(self, chat_id: str) -> None:
        from ..trust.registry import RegistryCommitUncertainError

        try:
            self.registry.bind_provision_chat(self.id, chat_id)
        except Exception as exc:
            logger.exception("failed to bind remote chat to provision %s", self.id)
            if isinstance(exc, RegistryCommitUncertainError) and exc.committed:
                self.record(chat_id, "registry_bind_uncertain")
                raise _Abort(
                    "⚠️ 受管群远端绑定结果不确定，已保留飞书群且不会执行删除；服务恢复核对后可安全重试。"
                ) from exc
            self.retain(chat_id)
            raise _Abort(
                "⚠️ 受管群远端绑定持久化失败；已保留该群但不授予信任，请由 Owner 处理。"
            ) from exc

    def acquire_chat(
        self,
        recovered_chat_id: str | None,
        *,
        name: str,
        description: str,
        sender_open_id: str,
    ) -> tuple[str, tuple[str, str] | None]:
        residual = self.pm.managed_group_residual(self.id)
        recovering: tuple[str, str] | None = None
        if residual and residual[1] == "untrusted_retained":
            validation = self.remote.validate_managed_chat(residual[0], self.owner_id)
            if validation is ManagedChatValidation.UNKNOWN:
                raise _Abort(
                    "⚠️ 无法确认上次保留群的 Owner/接收 Bot 身份，已继续阻止新建。"
                )
            if validation is ManagedChatValidation.INVALID:
                if not self.abandon():
                    raise _Abort("⚠️ Registry 创建意图未能持久清理，已失败关闭。")
                self.consume(residual)
                raise _Abort("⚠️ 上次保留群身份校验失败，恢复记录已清理。")
            self._bind_remote(residual[0])
            recovered_chat_id, recovering = residual[0], residual
        elif residual and residual[1] not in {
            "create_outcome_unknown",
            "registry_bind_uncertain",
        }:
            raise _Abort("⚠️ 上次建群留下未确认的飞书群，已阻止重复建群。")

        if recovered_chat_id:
            if (
                self.remote.validate_managed_chat(recovered_chat_id, self.owner_id)
                is not ManagedChatValidation.VALID
            ):
                self.record(recovered_chat_id, "recovered_chat_invalid")
                raise _Abort(
                    "⚠️ 无法确认恢复群仍存在且 Owner/接收 Bot 均在群内，已停止激活。"
                )
            return recovered_chat_id, recovering

        try:
            dispatch = self.registry.prepare_create_dispatch(
                self.id, dispatched_at=datetime.now(UTC)
            )
        except Exception as exc:
            logger.exception("managed group dispatch preparation failed")
            raise _Abort("⚠️ 建群派发状态无法确认，已阻止重复建群。") from exc
        if not dispatch:
            self.record("outcome_unknown", "create_outcome_unknown")
            raise _Abort("⚠️ 上次建群结果仍不确定，已阻止重复建群。")

        try:
            result = self.remote.create_chat(
                name=name,
                description=description,
                user_id_list=[sender_open_id],
                operation_id=self.id,
            )
        except CreateChatError as exc:
            if exc.disposition is CreateChatFailureDisposition.OUTCOME_UNKNOWN:
                try:
                    self.registry.mark_create_outcome_unknown(self.id)
                except Exception:
                    logger.exception("failed to mark unknown create outcome")
                self.record("outcome_unknown", "create_outcome_unknown")
            else:
                self.abandon()
            raise _Abort(f"❌ 建群失败: {exc}") from exc
        self._bind_remote(result.chat_id)
        return result.chat_id, None


class ProjectChatService:
    """Orchestrate `/new-chat` without exposing intermediate choices."""

    def __init__(
        self,
        project_manager: ProjectManager,
        lark_chat_client: LarkChatClient,
        reply_fn: Callable[[str, str, Optional[str]], object],
        send_to_chat_fn: Callable[[str, str, str, Optional[str]], object],
        managed_group_registry: Optional["ManagedGroupRegistry"] = None,
        owner_id: str = "",
        receiving_bot_ref: str = "",
    ) -> None:
        self._pm = project_manager
        self._lark = lark_chat_client
        self._reply = reply_fn
        self._send_to_chat = send_to_chat_fn
        self._managed_groups = managed_group_registry
        self._owner_id = owner_id
        self._receiving_bot_ref = receiving_bot_ref

    def handle(
        self,
        message_id: str,
        chat_id: str,
        sender_open_id: str,
        data: dict,
    ) -> None:
        settings = get_settings()
        path = os.path.expanduser(os.path.abspath(data.get("path") or os.getcwd()))
        name = data.get("name") or os.path.basename(os.path.normpath(path))
        suffix = data.get("suffix") or settings.project_chat_suffix
        for label, part in (("项目名", name), ("后缀", suffix)):
            if error := _name_error(part):
                self._reply(message_id, f"❌ {label}无效: {error}", None)
                return
        if (
            self._managed_groups is None
            or not self._owner_id
            or not self._receiving_bot_ref
        ):
            self._reply(message_id, "❌ 受管群服务未就绪，已保持失败关闭。", None)
            return

        lock = _creation_lock(path)
        if not lock.acquire(timeout=5):
            self._reply(message_id, "⏳ 正在处理中，请稍后再试", None)
            return
        try:
            self._handle_locked(
                message_id, chat_id, sender_open_id, name, suffix, path
            )
        except _Abort as exc:
            self._reply(message_id, str(exc), None)
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
        registry = self._managed_groups
        assert registry is not None
        ctx = self._pm.find_project_by_path(path, chat_id=None)
        same_name = self._pm.find_project_by_name(name, chat_id=None)
        if ctx is None and same_name is not None:
            raise _Abort("❌ 同名项目已绑定其他目录，请使用不同项目名。")

        provision_id = (
            f"new-chat:{self._owner_id}:{ProjectManager.generate_id(name)}:"
            f"{os.path.normpath(path)}"
        )
        provision = _Provision(
            self._pm, registry, self._lark, provision_id, self._owner_id
        )
        if ctx and ctx.bound_chat_id:
            self._verify_existing_binding(ctx, provision)
            if chat_id and chat_id not in ctx.allowed_chat_ids:
                ctx.add_chat_id(chat_id)
                if self._pm._save_projects() is False:
                    raise _Abort("❌ 更新项目可见范围失败，请重试")
            self._reply_jump_card(message_id, ctx)
            return

        intended_project_id = ctx.project_id if ctx else ProjectManager.generate_id(name)
        recovered = provision.begin(
            intended_project_id, path, self._receiving_bot_ref
        )
        group_name = f"{name.strip()}-{suffix.strip()}"
        new_chat_id, residual = provision.acquire_chat(
            recovered,
            name=group_name,
            description=self._description(name, path),
            sender_open_id=sender_open_id,
        )
        self._lark.add_managers(new_chat_id, [sender_open_id])
        ctx = self._bind_project(
            ctx,
            provision,
            chat_id,
            new_chat_id,
            group_name,
            name,
            path,
            residual,
        )
        self._reply_jump_card(message_id, ctx)
        self._send_welcome(new_chat_id, ctx)

    def _verify_existing_binding(
        self, ctx: ProjectContext, provision: _Provision
    ) -> None:
        registry = provision.registry
        try:
            active, grant = registry.trust_snapshot(ctx.bound_chat_id)
        except Exception as exc:
            raise _Abort("❌ 无法读取项目群信任事实，已保持失败关闭。") from exc
        if not active or not grant or (
            active.project_id != ctx.project_id
            or active.canonical_root_ref != ctx.root_path
            or active.origin is not ManagedGroupOrigin.GHOSTAP_CREATED
            or active.owner_id != self._owner_id
            or active.receiving_bot_ref != self._receiving_bot_ref
            or grant.project_id != ctx.project_id
            or grant.canonical_root_ref != ctx.root_path
            or grant.managed_group_id != ctx.bound_chat_id
            or grant.owner_id != self._owner_id
            or grant.backend_binding_ids
            or grant.connected_target_refs
        ):
            raise _Abort("❌ 现有项目群信任事实不完整，已保持失败关闭。")

        if self._pm.pending_managed_chat_binding_sagas_for_project(
            ctx.project_id, ctx.bound_chat_id
        ):
            saga = self._pm.resolve_managed_chat_binding_saga(
                project_id=ctx.project_id,
                chat_id=ctx.bound_chat_id,
                expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                expected_owner_id=self._owner_id,
                expected_receiving_bot_ref=self._receiving_bot_ref,
                expected_root_ref=ctx.root_path,
            )
            if saga is None or not (
                self._pm.validate_managed_chat_binding_saga(saga.operation_id)
                and self._pm.complete_managed_chat_binding_saga(saga.operation_id)
            ):
                raise _Abort("⚠️ 项目群本地绑定事务尚未安全收尾。")
        residual = self._pm.managed_group_residual(provision.id)
        if residual == (ctx.bound_chat_id, "untrusted_retained"):
            provision.consume(residual)

    def _bind_project(
        self,
        ctx: ProjectContext | None,
        provision: _Provision,
        source_chat_id: str,
        remote_chat_id: str,
        remote_chat_name: str,
        project_name: str,
        path: str,
        recovering_residual: tuple[str, str] | None,
    ) -> ProjectContext:
        from ..project.manager import ProjectCommitUncertainError
        from ..trust.registry import RegistryCommitUncertainError

        prepared = False
        try:
            if ctx is None:
                success, message, created = self._pm.create_project_with_managed_chat_saga(
                    project_id=ProjectManager.generate_id(project_name),
                    project_name=project_name,
                    root_path=path,
                    owner_chat_id=remote_chat_id,
                    additional_chat_id=source_chat_id,
                    managed_chat_id=remote_chat_id,
                    managed_chat_name=remote_chat_name,
                    created_at=time.time(),
                    operation_id=provision.id,
                    expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                    expected_owner_id=self._owner_id,
                    expected_receiving_bot_ref=self._receiving_bot_ref,
                )
                if not success or created is None:
                    provision.retain(remote_chat_id)
                    raise _Abort(f"⚠️ 创建项目失败: {message}；远端群未授信。")
                ctx = created
            else:
                bound, _ = self._pm.bind_managed_chat_for_saga(
                    ctx.project_id,
                    remote_chat_id,
                    chat_name=remote_chat_name,
                    created_at=time.time(),
                    operation_id=provision.id,
                    additional_chat_id=source_chat_id,
                    project_name=project_name,
                    expected_origin=ManagedGroupOrigin.GHOSTAP_CREATED,
                    expected_owner_id=self._owner_id,
                    expected_receiving_bot_ref=self._receiving_bot_ref,
                )
                if not bound:
                    raise OSError("project binding persistence failed")
            prepared = True
            provision.registry.activate(
                provision_id=provision.id,
                chat_id=remote_chat_id,
                project_id=ctx.project_id,
                canonical_root_ref=ctx.root_path,
            )
        except _Abort:
            raise
        except (RegistryCommitUncertainError, ProjectCommitUncertainError) as exc:
            if exc.committed:
                raise _Abort(
                    "⚠️ 项目群登记结果不确定，已保留群和绑定但不授予信任。"
                ) from exc
            self._compensate(provision, remote_chat_id, prepared)
            raise _Abort("❌ 项目群绑定失败，远端群已保留但未获得信任。") from exc
        except Exception as exc:
            logger.exception("project group binding failed")
            self._compensate(provision, remote_chat_id, prepared)
            raise _Abort("❌ 项目群绑定失败，远端群已保留但未获得信任。") from exc

        if not self._pm.complete_managed_chat_binding_saga(provision.id):
            raise _Abort("⚠️ 项目群已登记，但本地绑定事务收尾失败。")
        if recovering_residual is not None:
            provision.consume(recovering_residual)
        return ctx

    def _compensate(
        self, provision: _Provision, remote_chat_id: str, prepared: bool
    ) -> None:
        restored = (
            self._pm.restore_managed_chat_binding_saga(provision.id)
            if prepared
            else True
        )
        provision.retain(remote_chat_id)
        if not restored:
            self._pm.quarantine_bound_chat(remote_chat_id)

    def _description(self, name: str, path: str) -> str:
        lines = [f"🎯 项目: {name}", f"📁 目录: {path}"]
        try:
            result = subprocess.run(
                ["git", "-C", path, "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines.append(f"🔗 仓库: {result.stdout.strip()}")
        except Exception:
            pass
        lines.append("🤖 在群中直接对话即可进入 SMART 编程路由。")
        return "\n".join(lines)

    def _reply_jump_card(self, message_id: str, ctx: ProjectContext) -> None:
        from ..card.builders.project import ProjectBuilder

        msg_type, card_json = ProjectBuilder.build_project_chat_jump_card(ctx)
        self._reply(message_id, card_json, msg_type)

    def _send_welcome(self, chat_id: str, ctx: ProjectContext) -> None:
        text = (
            f"🎉 项目 **{ctx.project_name}** 专属群已就绪\n"
            f"📂 目录: `{ctx.root_path}`\n\n"
            "直接发送任务即可由 SMART 路由自动执行；也可使用显式工具或引擎命令。"
        )
        try:
            self._send_to_chat(chat_id, "text", text, None)
        except Exception as exc:
            logger.warning("send_welcome to %s failed: %s", chat_id[:12], exc)
