"""Pure, typed evaluation of GhostAP's configured security posture."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .settings import Settings


class IngressAccessMode(str, Enum):
    """How inbound users and chats are authorized."""

    ENFORCED = "enforced"
    SHADOW = "shadow"
    LEGACY_ALLOW_ALL = "legacy_allow_all"


class ShellAccessMode(str, Enum):
    """Explicit authorization profile for the host Shell lane."""

    DISABLED = "disabled"
    ADMIN_DM = "admin_dm"
    ALLOWLISTED = "allowlisted"
    ISOLATED = "isolated"
    TRUSTED_LOCAL = "trusted_local"


class SecuritySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class SecurityFinding:
    code: str
    severity: SecuritySeverity
    message: str


@dataclass(frozen=True)
class SecurityPosture:
    ingress_mode: IngressAccessMode
    shell_mode: ShellAccessMode
    employee_department_enabled: bool
    records_group_content: bool
    findings: tuple[SecurityFinding, ...]

    @property
    def is_valid(self) -> bool:
        return all(
            finding.severity is not SecuritySeverity.BLOCKING
            for finding in self.findings
        )


_ModeT = TypeVar("_ModeT", bound=Enum)


def _read_mode(
    settings: object,
    field_name: str,
    enum_type: type[_ModeT],
    default: _ModeT,
) -> _ModeT:
    """Read a Settings enum while tolerating lightweight test doubles."""

    raw = getattr(settings, field_name, default.value)
    if isinstance(raw, enum_type):
        return raw
    if not isinstance(raw, str):
        return default
    return enum_type(raw)


def _has_ids(settings: object, field_name: str) -> bool:
    raw = getattr(settings, field_name, ())
    return isinstance(raw, (str, tuple, list, set, frozenset)) and bool(raw)


def evaluate_security_posture(
    settings: Settings,
    *,
    isolation_ready: bool = False,
) -> SecurityPosture:
    """Return stable findings without mutating settings or probing the host."""

    ingress_mode = _read_mode(
        settings,
        "ingress_access_mode",
        IngressAccessMode,
        IngressAccessMode.ENFORCED,
    )
    shell_mode = _read_mode(
        settings,
        "shell_access_mode",
        ShellAccessMode,
        ShellAccessMode.DISABLED,
    )
    visible_employee_limit = getattr(
        settings,
        "autonomous_visible_employee_limit",
        0,
    )
    employee_enabled = (
        isinstance(visible_employee_limit, int)
        and not isinstance(visible_employee_limit, bool)
        and visible_employee_limit > 0
    )
    retention_days = getattr(
        settings,
        "employee_group_context_retention_days",
        30,
    )
    if not isinstance(retention_days, int) or isinstance(retention_days, bool):
        retention_days = 0

    findings: list[SecurityFinding] = []
    if ingress_mode is IngressAccessMode.LEGACY_ALLOW_ALL:
        findings.append(
            SecurityFinding(
                "ingress_legacy_allow_all",
                SecuritySeverity.WARNING,
                "legacy_allow_all is a time-bounded break-glass mode",
            )
        )
    if ingress_mode is IngressAccessMode.SHADOW:
        findings.append(
            SecurityFinding(
                "ingress_shadow_not_enforcing",
                SecuritySeverity.WARNING,
                "shadow records prospective denials but still allows traffic",
            )
        )
    if getattr(settings, "admin_bootstrap_scope", "p2p_only") == "any_chat":
        findings.append(
            SecurityFinding(
                "admin_bootstrap_any_chat",
                SecuritySeverity.WARNING,
                "first-admin bootstrap remains available from any chat",
            )
        )
    if (
        shell_mode is ShellAccessMode.ADMIN_DM
        and not _has_ids(settings, "admin_user_ids")
    ):
        findings.append(
            SecurityFinding(
                "shell_admin_missing",
                SecuritySeverity.BLOCKING,
                "admin_dm requires at least one configured administrator",
            )
        )
    if (
        shell_mode is ShellAccessMode.ALLOWLISTED
        and (
            not _has_ids(settings, "allowed_user_ids")
            or not _has_ids(settings, "allowed_chat_ids")
        )
    ):
        findings.append(
            SecurityFinding(
                "shell_allowlist_missing",
                SecuritySeverity.BLOCKING,
                "allowlisted requires both user and chat allowlists",
            )
        )
    if (
        shell_mode is ShellAccessMode.TRUSTED_LOCAL
        and getattr(settings, "shell_trusted_local_ack", False) is not True
    ):
        findings.append(
            SecurityFinding(
                "shell_trusted_local_unacknowledged",
                SecuritySeverity.BLOCKING,
                "trusted_local requires explicit risk acknowledgement",
            )
        )
    trusted_personal = (
        getattr(settings, "acp_trusted_personal_mode", False) is True
    )
    if trusted_personal and getattr(
        settings,
        "acp_trusted_personal_ack",
        False,
    ) is not True:
        findings.append(
            SecurityFinding(
                "acp_trusted_personal_unacknowledged",
                SecuritySeverity.BLOCKING,
                "ACP trusted-personal mode requires explicit risk acknowledgement",
            )
        )
    elif trusted_personal and not _has_ids(settings, "admin_user_ids"):
        findings.append(
            SecurityFinding(
                "acp_trusted_personal_admin_missing",
                SecuritySeverity.BLOCKING,
                "ACP trusted-personal mode requires a configured administrator",
            )
        )
    elif trusted_personal:
        findings.append(
            SecurityFinding(
                "acp_trusted_personal_active",
                SecuritySeverity.WARNING,
                "administrator project tasks may run ACP backends with full host and network access",
            )
        )
    if shell_mode is ShellAccessMode.ISOLATED and not isolation_ready:
        findings.append(
            SecurityFinding(
                "shell_isolation_unavailable",
                SecuritySeverity.BLOCKING,
                "isolated requires a successfully probed isolation backend",
            )
        )
    if employee_enabled and retention_days < 1:
        findings.append(
            SecurityFinding(
                "employee_group_retention_missing",
                SecuritySeverity.BLOCKING,
                "Employee Department requires bounded group-context retention",
            )
        )

    return SecurityPosture(
        ingress_mode=ingress_mode,
        shell_mode=shell_mode,
        employee_department_enabled=employee_enabled,
        records_group_content=employee_enabled,
        findings=tuple(findings),
    )
