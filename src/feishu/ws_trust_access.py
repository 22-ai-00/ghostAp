"""Project resolution, ingress access policy, and trust-zone checks for FeishuWSClient."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..project import ProjectContext
    from .ws_client import P2ImMessageReceiveV1

from ..access_control import (
    AccessDecision,
    AccessOperation,
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
    IngressAccessRequest,
    are_canonical_ingress_facts,
    build_ingress_access_policy,
)
from ..config import IngressAccessMode
from ..mode import PROGRAMMING_MODES
from ..trust.models import ActorKind, EffectiveTrust, TrustZone
from ..trust.registry import ManagedGroupRegistry
from ..trust.resolver import TrustZoneResolver
from .slash_command_parser import CommandMatch, SlashCommandParser

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("ghostap.audit")

_LARK_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_[0-9]+|@all")
_BOOTSTRAP_ACCESS_OPERATIONS = frozenset(
    {
        AccessOperation.BOOTSTRAP_HELP,
        AccessOperation.BOOTSTRAP_ADMIN,
    }
)


@dataclass(frozen=True, slots=True)
class _MentionedGroupCommand:
    """Parsed metadata for a uniquely bot-mentioned slash command in group chat."""

    tenant_key: str
    chat_id: str
    message_id: str
    bot_open_id: str


class WSTrustAccessMixin:
    """Mixin providing project resolution, ingress access policy, and trust-zone checks."""

    def _resolve_project_from_message(
        self, message_id: str, chat_id: str, parent_id: Optional[str] = None
    ) -> tuple[Optional[ProjectContext], Optional[str]]:
        """根据消息引用（parent/root）解析项目上下文。

        返回：
        - `project`: 最终解析到的 ProjectContext（或当前 active project）。
        - `auto_enter_mode`: 若该消息是回复某个编程会话/项目卡片，允许自动进入对应编程模式。
        """
        auto_enter_mode = None

        if parent_id:
            project_id = self._message_mapper.get_project_id(parent_id)
            if not isinstance(project_id, str):
                project_id = None
            if project_id:
                project = self._project_manager.get_project_for_chat(project_id, chat_id)
                if project:
                    self._project_manager.set_active_project(chat_id, project_id)
                    logger.info("通过消息引用切换到项目: %s", project.project_name)

                    # Resolve mode from ModeManager (single source of truth).
                    _proj_mode = self._mode_manager.get_mode(chat_id, project_id=project_id)
                    if _proj_mode in PROGRAMMING_MODES:
                        auto_enter_mode = _proj_mode.value

                    if auto_enter_mode:
                        logger.info("自动进入 %s 模式 (回复编程消息)", auto_enter_mode)

                    return project, auto_enter_mode

        bound_project = self._project_manager.find_by_bound_chat_id(chat_id)
        if bound_project is not None:
            return bound_project, None

        return self._project_manager.get_active_project(chat_id), None

    @staticmethod
    def _access_identifier_hash(value: str) -> str:
        """Hash a Feishu identifier for redacted audit logs."""
        return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]

    def _current_ingress_access_policy(self) -> IngressAccessPolicy:
        """Return the active immutable ingress access policy snapshot."""
        provider = getattr(self, "_ingress_access_policy_provider", None)
        if isinstance(provider, IngressAccessPolicyProvider):
            return provider.current
        # Direct unit construction historically bypasses _initialize. Keep that
        # path secure by deriving a fresh immutable snapshot from its settings.
        return build_ingress_access_policy(self.settings)

    def _decide_ingress_access(
        self,
        *,
        message_id: str,
        sender_id: str,
        chat_id: str,
        chat_type: str,
        command_match: CommandMatch | None,
        policy_snapshot: IngressAccessPolicy | None = None,
    ) -> AccessDecision:
        """Evaluate fail-closed ingress access policy and emit audit log entries."""
        try:
            policy = (
                policy_snapshot
                if policy_snapshot is not None
                else self._current_ingress_access_policy()
            )
            decision = policy.decide(
                IngressAccessRequest(
                    message_id=message_id,
                    sender_id=sender_id,
                    chat_id=chat_id,
                    chat_type=chat_type,
                    command_match=command_match,
                )
            )
        except Exception:
            logger.critical(
                "INGRESS_ACCESS_POLICY_EVALUATION_FAILED",
                exc_info=True,
            )
            policy = IngressAccessPolicy(
                admin_ids=frozenset(),
                allowed_user_ids=frozenset(),
                allowed_chat_ids=frozenset(),
                mode=IngressAccessMode.ENFORCED,
                admin_bootstrap_scope="p2p_only",
            )
            decision = AccessDecision(
                allowed=False,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code="access_policy_error",
                prospective_allowed=False,
            )

        if (
            not decision.allowed
            or decision.operation is AccessOperation.BOOTSTRAP_HELP
            or policy.mode.value in {
                "shadow",
                "legacy_allow_all",
            }
        ):
            audit_logger.warning(
                "INGRESS_ACCESS_DECISION mode=%s operation=%s reason=%s "
                "sender_hash=%s chat_hash=%s prospective_allowed=%s",
                policy.mode.value,
                decision.operation.value,
                decision.reason_code,
                self._access_identifier_hash(sender_id),
                self._access_identifier_hash(chat_id),
                decision.prospective_allowed,
            )
        return decision

    def _resolve_effective_trust(
        self,
        *,
        sender_id: str,
        chat_id: str,
        chat_type: str,
        message_id: str = "",
    ) -> EffectiveTrust | None:
        """Resolve current Registry trust, or ``None`` for legacy test wiring."""
        del message_id
        registry = getattr(self, "_managed_group_registry", None)
        if type(registry) is not ManagedGroupRegistry:
            return None
        try:
            groups = registry.managed_groups()
            group = next(
                (record for record in groups if record.chat_id == chat_id),
                None,
            )
            if group is not None:
                current_group, current_grant = registry.trust_snapshot(chat_id)
                groups = (() if current_group is None else (current_group,))
                grants = (() if current_grant is None else (current_grant,))
            else:
                grants = ()
            resolver = TrustZoneResolver(
                owner_id=getattr(self, "_managed_group_owner_id", ""),
                managed_groups=groups,
                project_grants=grants,
            )
        except Exception:
            logger.critical(
                "MANAGED_INGRESS_REGISTRY_SNAPSHOT_FAILED",
                exc_info=True,
            )
            resolver = TrustZoneResolver(
                owner_id=getattr(self, "_managed_group_owner_id", ""),
                managed_groups=(),
                project_grants=(),
            )
        return resolver.resolve(
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
        )

    @staticmethod
    def _managed_trust_access_decision(
        trust: EffectiveTrust | None,
    ) -> AccessDecision | None:
        """Return an authoritative decision for group trust zones.

        Owner P2P and clients without the durable Registry retain the existing
        access-policy path.  Managed actors bypass legacy enrollment; unknown
        group actors are denied before content parsing.
        """

        if trust is None:
            return None
        if trust.zone is TrustZone.OWNER_P2P:
            return AccessDecision(
                allowed=True,
                operation=AccessOperation.NORMAL_MESSAGE,
                reason_code="owner_p2p_trust",
                prospective_allowed=True,
                effective_trust=trust,
            )
        allowed = (
            trust.zone is TrustZone.MANAGED_AGENT_GROUP
            and trust.actor is ActorKind.OWNER
        )
        return AccessDecision(
            allowed=allowed,
            operation=AccessOperation.NORMAL_MESSAGE,
            reason_code=("managed_trust" if allowed else "external_or_unknown"),
            prospective_allowed=allowed,
            effective_trust=trust,
        )

    def _message_trust_access_decision(
        self,
        trust: EffectiveTrust | None,
        *,
        chat_type: str,
    ) -> AccessDecision | None:
        """Let open ingress and first-run P2P use the access policy."""

        policy = self._current_ingress_access_policy()
        if (
            trust is not None
            and trust.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP
            and policy.mode is IngressAccessMode.LEGACY_ALLOW_ALL
        ):
            return None
        if (
            chat_type == "p2p"
            and not policy.admin_ids
        ):
            return None
        return self._managed_trust_access_decision(trust)

    def _managed_ingress_action_allowed(
        self,
        trust: EffectiveTrust | None,
        *,
        text: str,
        command_match: CommandMatch | None,
    ) -> bool:
        """Return whether the resolved trust zone may execute an ingress action."""
        if trust is None:
            return True
        del text, command_match
        if (
            trust.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP
            and self._current_ingress_access_policy().mode
            is IngressAccessMode.LEGACY_ALLOW_ALL
        ):
            return True
        return trust.zone is not TrustZone.EXTERNAL_OR_UNKNOWN_GROUP

    def _current_trust_can_dispatch(
        self,
        trust: EffectiveTrust | None,
        *,
        project=None,
    ) -> bool:
        """Re-validate that trust revisions and project bindings have not drifted."""
        if trust is None:
            return True
        if (
            trust.zone is TrustZone.EXTERNAL_OR_UNKNOWN_GROUP
            and self._current_ingress_access_policy().mode
            is IngressAccessMode.LEGACY_ALLOW_ALL
        ):
            return True
        current_group_revision = None
        current_grant_revision = None
        if trust.managed_group is not None:
            registry = getattr(self, "_managed_group_registry", None)
            if type(registry) is not ManagedGroupRegistry:
                return False
            current_group, current_grant = registry.trust_snapshot(
                trust.managed_group.chat_id
            )
            current_group_revision = (
                current_group.revision if current_group is not None else None
            )
            current_grant_revision = (
                current_grant.revision if current_grant is not None else None
            )
            if (
                project is not None
                and (
                    current_group is None
                    or getattr(project, "project_id", None) != current_group.project_id
                    or getattr(project, "root_path", None)
                    != current_group.canonical_root_ref
                )
            ):
                return False
        if trust.zone is TrustZone.OWNER_P2P:
            return (
                trust.actor is ActorKind.OWNER
                and trust.managed_group is None
                and trust.group_revision is None
                and trust.grant_revision is None
                and current_group_revision is None
                and current_grant_revision is None
            )
        return (
            trust.zone is TrustZone.MANAGED_AGENT_GROUP
            and trust.actor is ActorKind.OWNER
            and trust.managed_group is not None
            and current_group_revision == trust.group_revision
            and current_grant_revision == trust.grant_revision
        )

    @staticmethod
    def _extract_canonical_ingress_facts(
        data: P2ImMessageReceiveV1,
    ) -> tuple[str, str, str, str] | None:
        """Read and validate the event trust root without parsing content."""

        try:
            message = data.event.message
            message_id = message.message_id
            chat_id = message.chat_id
            chat_type = message.chat_type
            sender_id = data.event.sender.sender_id.open_id
        except (AttributeError, TypeError):
            return None
        if not are_canonical_ingress_facts(
            message_id=message_id,
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
        ):
            return None
        return message_id, chat_id, chat_type, sender_id

    def _parse_uniquely_mentioned_group_command(
        self,
        data: P2ImMessageReceiveV1,
    ) -> _MentionedGroupCommand | None:
        """Parse one uniquely mentioned group slash without consulting runtime state."""

        try:
            message = data.event.message
            event_tenant_key = data.header.tenant_key
        except (AttributeError, TypeError):
            return None
        chat_id = getattr(message, "chat_id", None)
        if message.chat_type != "group" or message.message_type != "text":
            return None
        mentions = getattr(message, "mentions", None)
        if not isinstance(mentions, (list, tuple)) or len(mentions) != 1:
            return None
        mention = mentions[0]
        identity = getattr(mention, "id", None)
        mention_key = getattr(mention, "key", None)
        target_open_id = getattr(identity, "open_id", None)
        mention_tenant_key = getattr(mention, "tenant_key", None)
        if (
            not isinstance(event_tenant_key, str)
            or not event_tenant_key
            or not isinstance(chat_id, str)
            or not chat_id
            or mention_tenant_key != event_tenant_key
            or not isinstance(mention_key, str)
            or not mention_key.startswith("@_user_")
            or not mention_key.removeprefix("@_user_").isdigit()
            or not isinstance(target_open_id, str)
            or not target_open_id.startswith("ou_")
        ):
            return None
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            return None
        try:
            decoded = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        raw_text = decoded.get("text") if isinstance(decoded, dict) else None
        if (
            not isinstance(raw_text, str)
            or _LARK_MENTION_PLACEHOLDER_RE.findall(raw_text) != [mention_key]
        ):
            return None
        addressed = raw_text.lstrip()
        if not addressed.startswith(mention_key):
            return None
        command_text = addressed[len(mention_key) :]
        if not command_text[:1].isspace():
            return None
        if SlashCommandParser.parse(command_text.lstrip()) is None:
            return None
        message_id = getattr(message, "message_id", None)
        if not isinstance(message_id, str) or not message_id:
            return None
        return _MentionedGroupCommand(
            tenant_key=event_tenant_key,
            chat_id=chat_id,
            message_id=message_id,
            bot_open_id=target_open_id,
        )
