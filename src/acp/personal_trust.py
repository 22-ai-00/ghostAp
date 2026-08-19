"""Task-scoped trusted-personal permission leases for ordinary ACP sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _configured_ids(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (tuple, list, set, frozenset)):
        return frozenset(
            str(item).strip() for item in value if str(item).strip()
        )
    return frozenset()


def trusted_personal_permissions_requested(
    settings: object,
    *,
    project: object | None,
    sender_id: str,
) -> bool:
    """Return whether this exact ordinary-programming task may bypass ACP policy.

    The global switch is deliberately insufficient on its own. The caller must
    be a configured administrator and the task must be bound to a concrete
    project root, keeping the escape hatch out of shared engine/employee paths.
    """

    if (
        getattr(settings, "acp_trusted_personal_mode", False) is not True
        or getattr(settings, "acp_trusted_personal_ack", False) is not True
    ):
        return False
    if not sender_id or sender_id not in _configured_ids(
        getattr(settings, "admin_user_ids", ())
    ):
        return False
    return bool(
        str(getattr(project, "project_id", "") or "").strip()
        and str(getattr(project, "root_path", "") or "").strip()
    )


@dataclass
class TrustedPersonalPermissionLease:
    """Enable broad ACP permissions only for the lifetime of one user task."""

    enabled: bool
    _sessions: dict[int, Any] = field(default_factory=dict, init=False)

    @staticmethod
    def _mark_dead(session: object) -> None:
        try:
            setattr(session, "_force_dead", True)
        except Exception:
            pass

    def acquire(self, session: object) -> bool:
        """Acquire this task's lease on one session, including replacements."""

        if not self.enabled:
            return False
        identity = id(session)
        if identity in self._sessions:
            return True
        setter = getattr(session, "set_trusted_personal_permissions", None)
        if not callable(setter):
            return False
        try:
            applied = setter(True)
        except Exception as exc:
            self._mark_dead(session)
            raise RuntimeError(
                "ACP 个人可信权限租约启用失败；会话已隔离"
            ) from exc
        if applied is not True:
            self._mark_dead(session)
            raise RuntimeError("ACP 个人可信权限租约启用失败；会话已隔离")
        self._sessions[identity] = session
        return True

    def release_all(self) -> tuple[object, ...]:
        """Revoke every acquired session; uncertain sessions become unusable."""

        failures: list[object] = []
        sessions = tuple(reversed(tuple(self._sessions.values())))
        self._sessions.clear()
        for session in sessions:
            if getattr(session, "_force_dead", False) is True:
                continue
            setter = getattr(session, "set_trusted_personal_permissions", None)
            try:
                revoked = callable(setter) and setter(False) is True
            except Exception:
                revoked = False
            if not revoked:
                self._mark_dead(session)
                failures.append(session)
        return tuple(failures)
