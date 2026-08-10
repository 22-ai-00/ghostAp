"""Typed command boundary between the main Bot and employee provisioning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class EmployeeHireRequest:
    employee_name: str
    tool: str
    model: str
    effort: str
    chat_id: str
    message_id: str
    requester_principal_id: str
    requester_union_id: str = ""
    tenant_key: str = ""
    profile: str = "standard"
    role: str = ""
    persona: str = ""
    personality_traits: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    existing_app_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "personality_traits", tuple(self.personality_traits))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "permissions", tuple(self.permissions))


@dataclass(frozen=True, slots=True)
class EmployeeProfileTemplate:
    role: str
    persona: str
    personality_traits: tuple[str, ...]
    capabilities: tuple[str, ...]
    permissions: tuple[str, ...]
    permission_profile: str


ALLOWED_PERSONALITY_TRAITS = frozenset(
    {
        "严谨",
        "注重细节",
        "主动沟通",
        "批判性思维",
        "追求质量",
        "细致",
        "追求覆盖",
        "全局视角",
        "有条理",
        "抽象思维",
        "系统设计",
        "表达清晰",
        "注重结构",
        "好奇",
        "证据导向",
    }
)
ALLOWED_CAPABILITIES = frozenset(
    {
        "coding",
        "testing",
        "review",
        "planning",
        "architecture",
        "writing",
        "research",
        "file_read",
        "file_write",
        "shell",
        "git",
        "vision",
        "attachments",
    }
)
ALLOWED_PERMISSIONS = frozenset({"file_read", "file_write", "shell", "git"})

_TOOL_DEFAULT_ROLE = {
    "traex": "coder",
    "codex": "coder",
    "aiden": "coder",
    "gemini": "coder",
    "claude": "reviewer",
    "coco": "writer",
}
_ROLE_PROFILES = {
    "coder": EmployeeProfileTemplate(
        role="coder",
        persona="以可靠、可验证的改动交付软件；先理解现有约束，再实现、测试并清楚交接风险。",
        personality_traits=("严谨", "注重细节", "主动沟通"),
        capabilities=(
            "coding",
            "testing",
            "review",
            "file_read",
            "file_write",
            "shell",
            "git",
        ),
        permissions=("file_read", "file_write", "shell", "git"),
        permission_profile="development",
    ),
    "reviewer": EmployeeProfileTemplate(
        role="reviewer",
        persona="独立检查事实、风险和验收证据；给出可执行的修订建议，不替他人掩盖不确定性。",
        personality_traits=("批判性思维", "追求质量", "证据导向"),
        capabilities=("review", "research", "file_read"),
        permissions=("file_read",),
        permission_profile="read_only",
    ),
    "tester": EmployeeProfileTemplate(
        role="tester",
        persona="从失败边界和用户契约出发设计验证，优先产出可重复的回归证据。",
        personality_traits=("细致", "追求覆盖", "证据导向"),
        capabilities=("testing", "review", "file_read", "shell"),
        permissions=("file_read", "shell"),
        permission_profile="testing",
    ),
    "planner": EmployeeProfileTemplate(
        role="planner",
        persona="把目标拆成有依赖、有验收条件的步骤，持续暴露假设、风险和未决问题。",
        personality_traits=("全局视角", "有条理", "主动沟通"),
        capabilities=("planning", "research", "file_read"),
        permissions=("file_read",),
        permission_profile="read_only",
    ),
    "architect": EmployeeProfileTemplate(
        role="architect",
        persona="维护系统边界和长期一致性，用明确合约约束实现并审查演进风险。",
        personality_traits=("抽象思维", "系统设计", "严谨"),
        capabilities=("architecture", "planning", "review", "file_read"),
        permissions=("file_read",),
        permission_profile="read_only",
    ),
    "writer": EmployeeProfileTemplate(
        role="writer",
        persona="把复杂信息整理成准确、清晰、可追溯的交付物，并主动核对读者目标。",
        personality_traits=("表达清晰", "注重结构", "主动沟通"),
        capabilities=("writing", "research", "file_read"),
        permissions=("file_read",),
        permission_profile="read_only",
    ),
}


def recommended_employee_profile(tool: str, role: str = "") -> EmployeeProfileTemplate:
    """Return the version-controlled least-privilege profile for a hire."""

    profile_role = role if role in _ROLE_PROFILES else _TOOL_DEFAULT_ROLE.get(tool, "planner")
    template = _ROLE_PROFILES[profile_role]
    return replace(template, role=role or template.role)


def _validate_profile_values(
    values: tuple[str, ...],
    *,
    field_name: str,
    allowlist: frozenset[str],
    maximum: int,
) -> tuple[str, ...]:
    if len(values) > maximum or len(set(values)) != len(values):
        raise ValueError(f"invalid employee profile {field_name}")
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value not in allowlist
        for value in values
    ):
        raise ValueError(f"invalid employee profile {field_name}")
    return values


def complete_employee_hire_request(request: EmployeeHireRequest) -> EmployeeHireRequest:
    """Fill empty profile fields and validate every bounded choice."""

    template = recommended_employee_profile(request.tool, request.role)
    role = request.role.strip() if isinstance(request.role, str) else ""
    persona = request.persona.strip() if isinstance(request.persona, str) else ""
    if len(role) > 80 or len(persona) > 600:
        raise ValueError("invalid employee profile text")
    traits = _validate_profile_values(
        request.personality_traits or template.personality_traits,
        field_name="personality_traits",
        allowlist=ALLOWED_PERSONALITY_TRAITS,
        maximum=6,
    )
    capabilities = _validate_profile_values(
        request.capabilities or template.capabilities,
        field_name="capabilities",
        allowlist=ALLOWED_CAPABILITIES,
        maximum=12,
    )
    permissions = _validate_profile_values(
        request.permissions or template.permissions,
        field_name="permissions",
        allowlist=ALLOWED_PERMISSIONS,
        maximum=4,
    )
    if not set(permissions).issubset(capabilities):
        raise ValueError("invalid employee profile permission capability binding")
    return replace(
        request,
        role=role or template.role,
        persona=persona or template.persona,
        personality_traits=traits,
        capabilities=capabilities,
        permissions=permissions,
    )


class EmployeeHireService(Protocol):
    pass


__all__ = [
    "ALLOWED_CAPABILITIES",
    "ALLOWED_PERMISSIONS",
    "ALLOWED_PERSONALITY_TRAITS",
    "EmployeeHireRequest",
    "EmployeeHireService",
    "EmployeeProfileTemplate",
    "complete_employee_hire_request",
    "recommended_employee_profile",
]
