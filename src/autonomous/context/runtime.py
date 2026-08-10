"""Production-only authority adapters for employee Context composition."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..authorization import EmployeeAuthorizationScope
from ..provisioning.hire_state import HirePhase
from .models import AuthorizedContextRequest


class RuntimeRequesterChatAcl:
    """Require an authenticated requester allowlist in addition to membership."""

    def __init__(
        self,
        *,
        allowed_requesters: Iterable[str],
        allowed_chats: Iterable[str] = (),
    ) -> None:
        self._requesters = frozenset(allowed_requesters)
        self._chats = frozenset(allowed_chats)

    @property
    def configured(self) -> bool:
        return bool(self._requesters)

    def is_authorized(self, request: AuthorizedContextRequest) -> bool:
        return (
            isinstance(request, AuthorizedContextRequest)
            and request.requester_principal_id in self._requesters
            and (
                request.authorization_scope
                is EmployeeAuthorizationScope.OWNER_P2P
                or not self._chats
                or request.chat_id in self._chats
            )
        )


class RuntimeEmployeeGenerationAuthority:
    """Bind Context requests to the durable ACTIVE hire and live Channel."""

    def __init__(
        self,
        *,
        hire_service_provider: Callable[[], Any],
        channel_supervisor: Any,
        data_composition: Any,
    ) -> None:
        self._hire_service_provider = hire_service_provider
        self._channels = channel_supervisor
        self._data = data_composition

    def is_current(self, request: AuthorizedContextRequest) -> bool:
        try:
            self._data.service.rebuild_projection()
            service = self._hire_service_provider()
            service.synchronize_projection()
            state = next(item for item in service.list_states() if item.agent_id == request.agent_id)
            if (
                state.phase is not HirePhase.ACTIVE
                or state.tenant_key != request.tenant_key
                or state.bot_principal_id != request.bot_principal_id
                or state.app_id != request.app_id
                or state.channel_generation != request.channel_generation
                or (
                    request.authorization_scope
                    is EmployeeAuthorizationScope.OWNER_P2P
                    and state.requester_principal_id
                    != request.requester_principal_id
                )
            ):
                return False
            status = self._channels.status(request.agent_id)
            status_state = getattr(status, "state", None)
            status_value = getattr(status_state, "value", status_state)
            return (
                status_value == "ready"
                and getattr(status, "generation", None) == request.channel_generation
                and getattr(status, "identity", {}).get("app_id") == request.app_id
                and getattr(status, "ready_metadata", {}).get("connection_id") == state.channel_connection_id
            )
        except Exception:
            return False


def parse_requester_acl(settings: Any) -> RuntimeRequesterChatAcl:
    manager_acl = getattr(settings, "autonomous_manager_acl", "")
    if isinstance(manager_acl, str):
        managers = {item.strip() for item in manager_acl.split(",") if item.strip()}
    else:
        managers = set(manager_acl or ())
    managers.update(getattr(settings, "admin_user_ids", ()) or ())
    return RuntimeRequesterChatAcl(
        allowed_requesters=managers,
        allowed_chats=getattr(settings, "allowed_chat_ids", ()) or (),
    )


__all__ = [
    "RuntimeEmployeeGenerationAuthority",
    "RuntimeRequesterChatAcl",
    "parse_requester_acl",
]
