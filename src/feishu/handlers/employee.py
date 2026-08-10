"""Read-only Employee Department command handler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...autonomous.provisioning.hire_port import EmployeeRoleUpdateRequest
from ...card.builders.system import SystemBuilder
from ...config import get_settings
from ...thread import get_current_sender_id
from .base import BaseHandler

if TYPE_CHECKING:
    from ...project import ProjectContext


logger = logging.getLogger(__name__)

_ROSTER_UNAVAILABLE = "数字员工目录暂不可用，请稍后重试。"
_ROLE_USAGE = "用法：`/employee-role <员工名> <职责>`"
_ROLE_UNAVAILABLE = "数字员工职责更新暂不可用，请稍后重试。"


class EmployeeHandler(BaseHandler):
    """Expose tenant-scoped, Journal-backed employee read models."""

    @staticmethod
    def _display_entry(employee: Any) -> dict[str, object]:
        state = getattr(employee, "state", "")
        state_value = getattr(state, "value", state)
        groups = getattr(employee, "member_groups", ())
        if not isinstance(groups, (tuple, list, set, frozenset)):
            groups = ()
        return {
            "agent_id": getattr(employee, "agent_id", ""),
            "name": getattr(employee, "name", ""),
            "emoji": getattr(employee, "emoji", "🤖"),
            "state": state_value,
            "role": getattr(employee, "role", ""),
            "tool": getattr(employee, "tool", ""),
            "model": getattr(employee, "model", ""),
            "profile": getattr(employee, "profile", ""),
            "effort": getattr(employee, "effort", ""),
            "group_count": len(groups),
            "created_at": getattr(employee, "created_at", 0.0),
        }

    def list_employees_roster(
        self,
        message_id: str,
        chat_id: str,
        project: "ProjectContext | None" = None,
    ) -> None:
        """Render every visible employee from the caller's tenant projection."""

        del chat_id, project
        resolver = self.ctx.tenant_key_resolver
        service = self.ctx.employee_hire_service
        list_roster = getattr(service, "list_employee_roster", None)
        if not callable(resolver) or not callable(list_roster):
            self.reply_text(message_id, _ROSTER_UNAVAILABLE)
            return

        try:
            tenant_key = resolver()
            if not isinstance(tenant_key, str) or not tenant_key.strip():
                raise RuntimeError("employee roster tenant is unavailable")
            employees = list_roster(tenant_key.strip(), include_archived=False)
            entries = [self._display_entry(employee) for employee in employees]
            _msg_type, card_content = SystemBuilder.build_employee_roster_card(entries)
        except Exception as exc:
            logger.error(
                "employee roster projection unavailable: %s",
                type(exc).__name__,
            )
            self.reply_text(message_id, _ROSTER_UNAVAILABLE)
            return

        self.reply_card(message_id, card_content)

    @staticmethod
    def _admin_ids() -> frozenset[str]:
        try:
            raw = getattr(get_settings(), "admin_user_ids", frozenset())
        except Exception:
            return frozenset()
        if isinstance(raw, str):
            return frozenset(item.strip() for item in raw.split(",") if item.strip())
        if isinstance(raw, (set, frozenset, tuple, list)):
            return frozenset(item for item in raw if isinstance(item, str) and item)
        return frozenset()

    def update_employee_role(
        self,
        message_id: str,
        chat_id: str,
        args: str,
    ) -> None:
        """Update one tenant employee role through the Journal-backed service."""

        del chat_id
        sender_id = get_current_sender_id() or ""
        if not sender_id or sender_id not in self._admin_ids():
            self.reply_text(message_id, "仅 GhostAP 管理员可以设置数字员工职责。")
            return

        parts = (args or "").strip().split(maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            self.reply_text(message_id, _ROLE_USAGE)
            return
        employee_name, role = (part.strip() for part in parts)

        resolver = self.ctx.tenant_key_resolver
        service = self.ctx.employee_hire_service
        update_role = getattr(service, "update_employee_role", None)
        if not callable(resolver) or not callable(update_role):
            self.reply_text(message_id, _ROLE_UNAVAILABLE)
            return
        try:
            tenant_key = resolver()
        except Exception as exc:
            logger.error(
                "employee role tenant resolution failed: %s",
                type(exc).__name__,
            )
            self.reply_text(message_id, _ROLE_UNAVAILABLE)
            return
        if not isinstance(tenant_key, str) or not tenant_key.strip():
            self.reply_text(message_id, _ROLE_UNAVAILABLE)
            return
        try:
            updated = update_role(
                EmployeeRoleUpdateRequest(
                    tenant_key=tenant_key.strip(),
                    employee=employee_name,
                    role=role,
                    requester_principal_id=sender_id,
                    message_id=message_id,
                )
            )
        except Exception as exc:
            logger.error(
                "employee role update failed: %s",
                type(exc).__name__,
            )
            self.reply_text(
                message_id,
                "职责更新失败：请确认员工存在、职责有效且当前状态允许修改。",
            )
            return

        display_name = (
            getattr(updated, "name", "")
            or getattr(updated, "employee_name", "")
            or employee_name
        )
        display_role = getattr(updated, "role", "") or role
        self.reply_text(
            message_id,
            f"✅ 已将数字员工 **{display_name}** 的职责更新为：{display_role}",
        )


__all__ = ["EmployeeHandler"]
