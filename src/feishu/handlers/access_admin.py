"""Admin bootstrap, access control, and permission hint helpers for SystemHandler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...project import ProjectContext
    from ..slash_command_parser import CommandMatch

logger = logging.getLogger(__name__)


def _system_module():
    """Return the system handler module for runtime-patchable symbols."""
    from . import system as _sys_mod

    return _sys_mod


class AccessAdminMixin:
    """Mixin providing /setadmin, /access, and slash permission notices."""

    def _handle_retired_autonomous_manager_command(
        self,
        message_id: str,
        chat_id: str,
        m: "CommandMatch",
        project: "Optional[ProjectContext]" = None,
    ) -> None:
        """Fail closed for the retired standalone Autonomous Manager surface."""
        del chat_id, project
        if m.command == "/goal" and m.args:
            self.reply_text(
                message_id,
                _system_module().UI_TEXT["system_autonomous_manager_retired_goal"].format(task=m.args),
            )
            return
        self.reply_text(message_id, _system_module().UI_TEXT["system_autonomous_manager_retired"])

    def _handle_setadmin_command(self, message_id: str, chat_id: str, args: str = "") -> None:
        """Handle /setadmin in private chat or reject unauthorized/group usage."""
        from ...thread import get_current_is_p2p, get_current_sender_id

        sender_id = get_current_sender_id() or ""
        chat_type = "p2p" if get_current_is_p2p() else "group"
        result = self._admin_bootstrap_service().set_admin(
            sender_id,
            args,
            chat_type=chat_type,
            chat_id=chat_id,
            message_id=message_id,
        )
        if result.success:
            if result.code == "bootstrap":
                self.reply_text(message_id, _system_module().UI_TEXT["system_setadmin_bootstrap_success"])
            else:
                self.reply_text(
                    message_id,
                    _system_module().UI_TEXT["system_setadmin_update_success"].format(admin_id=result.target_id),
                )
            return

        if self._reply_commit_outcome_error(
            message_id,
            result.code,
            target_label="管理员配置",
            policy_unchanged_msg="在线访问策略未改变。",
            refresh_failed_msg="当前进程仍使用旧策略，请修复配置后重启。",
            denied_msg=_system_module().UI_TEXT["system_setadmin_denied"],
            prechecks={
                "missing_sender": _system_module().UI_TEXT["system_setadmin_missing_sender"],
                "invalid_target": _system_module().UI_TEXT["system_setadmin_invalid_target"],
                "bootstrap_requires_p2p": _system_module().UI_TEXT["system_setadmin_requires_p2p"],
                "rate_limited": _system_module().UI_TEXT["system_setadmin_rate_limited"],
            },
        ):
            return

    def _reply_commit_outcome_error(
        self,
        message_id: str,
        code: str,
        *,
        target_label: str,
        policy_unchanged_msg: str,
        refresh_failed_msg: str,
        denied_msg: str,
        prechecks: dict[str, str],
    ) -> bool:
        if code in prechecks:
            self.reply_error(message_id, prechecks[code])
            return True
        commit_errors = {
            "persistence_failed": f"{target_label}持久化失败，{policy_unchanged_msg}",
            "commit_uncertain": (
                f"{target_label}文件替换已发生，但目录耐久性确认失败；"
                "在线策略已按磁盘现状重新对齐，并记录了安全阻断项。"
            ),
            "commit_uncertain_unreconciled": (
                f"{target_label}替换状态不确定，且无法读取磁盘现状；"
                "在线策略保持原快照，已记录安全阻断项。"
            ),
            "commit_uncertain_refresh_failed": (
                f"{target_label}替换状态不确定；已读取磁盘现状，但运行时刷新失败，"
                "并已记录安全阻断项。"
            ),
            "commit_cleanup_failed": (
                f"{target_label}已持久化并在线生效，但事务资源清理失败；"
                "已记录安全阻断项，请检查文件系统状态。"
            ),
            "commit_cleanup_refresh_failed": (
                f"{target_label}已持久化，但事务资源清理和运行时刷新均失败；"
                "已记录安全阻断项，请修复后重启。"
            ),
            "policy_refresh_failed": (
                f"{target_label}已写入磁盘，但在线访问策略刷新失败；"
                f"{refresh_failed_msg}"
            ),
            "settings_mirror_failed": (
                f"{target_label}已写入磁盘，但运行时设置镜像失败；"
                "在线策略保持旧快照并已记录安全阻断项。"
            ),
        }
        msg = commit_errors.get(code)
        if msg is not None:
            self.reply_error(message_id, msg)
        else:
            self.reply_text(message_id, denied_msg)
        return True

    def _handle_access_command(
        self,
        message_id: str,
        chat_id: str,
        args: str = "",
    ) -> None:
        """Handle owner-only /access policy inspection and update commands."""
        from ...thread import get_current_is_p2p, get_current_sender_id

        access_args = (args or "").strip()
        if access_args.casefold().startswith("rotate-main-bot "):
            sender_id = get_current_sender_id() or ""
            owner_id = vars(self.ctx).get("managed_group_owner_id", "")
            rotate = vars(self.ctx).get("managed_group_bot_rotation")
            expected_bot_ref = access_args.split(maxsplit=1)[1].strip()
            if (
                not get_current_is_p2p()
                or not owner_id
                or sender_id != owner_id
                or not callable(rotate)
                or not expected_bot_ref
            ):
                self.reply_error(message_id, "仅配置的 Owner 可在私聊中执行主 Bot 轮换。")
                return
            rotated, rejected = rotate(expected_bot_ref)
            self.reply_text(
                message_id,
                f"主 Bot 轮换完成：已更新 {rotated} 个群，远端校验未通过 {rejected} 个群。",
            )
            return

        if access_args.casefold() == "migration-status":
            sender_id = get_current_sender_id() or ""
            owner_id = vars(self.ctx).get("managed_group_owner_id", "")
            registry = vars(self.ctx).get("managed_group_registry")
            if (
                not get_current_is_p2p()
                or not owner_id
                or sender_id != owner_id
                or registry is None
            ):
                self.reply_error(message_id, "仅配置的 Owner 可在私聊中查看迁移待办。")
                return
            dispositions = registry.migration_dispositions()
            if not dispositions:
                self.reply_text(message_id, "✅ 当前没有待人工处理的受管群迁移项。")
                return
            lines = ["受管群迁移待办："]
            lines.extend(
                f"- `{chat}` → `{project}`：{status}"
                for chat, project, status in dispositions
            )
            self.reply_text(message_id, "\n".join(lines))
            return

        if access_args.casefold().startswith("adopt-chat "):
            parts = access_args.split(maxsplit=2)
            sender_id = get_current_sender_id() or ""
            owner_id = vars(self.ctx).get("managed_group_owner_id", "")
            if not get_current_is_p2p() or not owner_id or sender_id != owner_id:
                self.reply_error(
                    message_id,
                    "仅配置的 Owner 可在与 Bot 的私聊中收养已有群。",
                )
                return
            if (
                len(parts) != 3
                or not parts[1].startswith("oc_")
                or not parts[2].strip()
            ):
                self.reply_error(
                    message_id,
                    "用法：`/access adopt-chat <oc_chat_id> <project-id-or-exact-name>`",
                )
                return
            project_handler = self.ctx.handlers.get("project")
            adopt = getattr(project_handler, "adopt_managed_chat", None)
            if not callable(adopt):
                self.reply_error(message_id, "受管群收养服务未就绪。")
                return
            adopt(message_id, parts[1], parts[2].strip())
            return

        if access_args.casefold() != "allow-chat":
            self.reply_text(
                message_id,
                "用法：目标群内发送 `/access allow-chat`，或由 Owner 私聊发送 "
                "`/access adopt-chat <oc_chat_id> <project-id-or-exact-name>`。",
            )
            return

        sender_id = get_current_sender_id() or ""
        chat_type = "p2p" if get_current_is_p2p() else "group"
        result = self._admin_bootstrap_service().allow_current_chat(
            sender_id,
            chat_id,
            chat_type=chat_type,
            message_id=message_id,
        )
        if result.success:
            self.reply_text(message_id, "✅ 当前群已加入访问白名单，立即生效。")
            return
        self._reply_commit_outcome_error(
            message_id,
            result.code,
            target_label="群授权",
            policy_unchanged_msg="在线访问策略未改变。",
            refresh_failed_msg="当前进程仍拒绝该群，请修复配置后重启。",
            denied_msg="当前账号无权授权此群。",
            prechecks={
                "access_requires_group": "请在需要授权的目标群内发送 `/access allow-chat`。",
            },
        )

    def _admin_bootstrap_service(self):
        """Construct the single-admin bootstrap service bound to .env storage."""
        from ...admin_bootstrap import AdminBootstrapService

        return AdminBootstrapService(
            settings_getter=lambda: self.settings,
            policy_provider=self.ctx.ingress_access_policy_provider,
            env_store=self.ctx.ingress_env_store,
        )

    def _slash_permission_hint_text(self) -> str:
        """Return a user-safe optional discovery hint without exposing errors."""

        provider = getattr(self.ctx, "slash_permission_provider", None)
        if not callable(provider):
            return ""
        try:
            required = provider()
        except Exception:
            logger.warning("Slash permission hint provider failed", exc_info=True)
            return ""
        url = getattr(required, "authorization_url", "")
        if not isinstance(url, str) or not url.startswith(
            "https://open.feishu.cn/app/"
        ):
            return ""
        return (
            "⚠️ `/` 命令发现面板尚未开通可选权限，"
            "但不开通不影响消息、Shell 和编程工具使用。\n"
            "点击此链接进行权限申请并发布，GhostAP 会自动恢复：\n"
            f"{url}"
        )

    def _reply_slash_permission_hint(self, message_id: str) -> Optional[str]:
        """Reply in thread/message when the user invokes /slash permission guidance."""
        hint = self._slash_permission_hint_text()
        if not hint:
            return None
        return self.reply_text(message_id, hint)

    def show_bootstrap_help(self, message_id: str, chat_id: str) -> Optional[str]:
        """Reply to unauthenticated P2P help without leaking project context."""

        del chat_id
        text = (
            "GhostAP 已启动，但当前尚未配置管理员，因此普通命令仍按入口访问规则拒绝。\n"
            "请在当前私聊发送 `/setadmin` 完成首次管理员设置；设置成功后可使用"
            " `/help`、`/codex`、`/dsh` 等命令。"
        )
        hint = self._slash_permission_hint_text()
        if hint:
            text = f"{text}\n\n{hint}"
        return self.reply_text(message_id, text)
