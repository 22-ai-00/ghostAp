"""Read-only Employee Department command handler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...card.builders.system import SystemBuilder
from .base import BaseHandler

if TYPE_CHECKING:
    from ...project import ProjectContext


logger = logging.getLogger(__name__)

_ROSTER_UNAVAILABLE = "数字员工目录暂不可用，请稍后重试。"


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
            employees = list_roster(tenant_key.strip())
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


__all__ = ["EmployeeHandler"]
