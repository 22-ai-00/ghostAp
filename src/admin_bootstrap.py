"""Durable admin and current-chat enrollment for ingress authorization."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from .access_control import (
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
    are_canonical_ingress_facts,
    build_ingress_access_policy,
    is_canonical_sender_id,
    normalize_access_ids,
)
from .config import get_settings
from .config.env_file_store import (
    AtomicEnvFileStore,
    EnvCommitUncertainError,
    EnvFileSnapshot,
    EnvPostCommitCleanupError,
    EnvPreReplaceError,
)

audit_logger = logging.getLogger("ghostap.audit")
logger = logging.getLogger(__name__)

_RATE_LIMIT_SECONDS = 60
_COMMIT_UNCERTAIN_FINDING = "ingress_env_commit_uncertain"
_CLEANUP_FAILED_FINDING = "ingress_env_post_commit_cleanup_failed"
_REFRESH_FAILED_FINDING = "ingress_policy_refresh_failed"
_SETTINGS_MIRROR_FAILED_FINDING = "ingress_settings_mirror_failed"


@dataclass(frozen=True)
class AdminBootstrapResult:
    success: bool
    code: str
    admin_id: str = ""
    target_id: str = ""


class _EnrollmentRejected(Exception):
    def __init__(self, code: str, *, target_id: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.target_id = target_id


class AdminBootstrapService:
    """Persist and immediately publish access-control enrollment changes.

    Every caller must supply the canonical message, sender, chat and chat-type
    facts already checked at WebSocket ingress. The service validates them
    again before rate limiting or touching persistence.
    """

    _global_lock = threading.Lock()  # leaf lock: never acquire LockLevel locks here
    _last_attempt: dict[str, float] = {}

    def __init__(
        self,
        *,
        env_path: str | os.PathLike[str] = ".env",
        settings_getter: Callable = get_settings,
        policy_provider: IngressAccessPolicyProvider | None = None,
        env_store: AtomicEnvFileStore | None = None,
    ) -> None:
        self._env_path = Path(env_path)
        self._settings_getter = settings_getter
        self._policy_provider = policy_provider
        self._env_store = env_store or AtomicEnvFileStore(self._env_path)

    def set_admin(
        self,
        sender_id: str,
        requested_target: str = "",
        chat_type: str = "",
        chat_id: str = "",
        message_id: str = "",
    ) -> AdminBootstrapResult:
        if not are_canonical_ingress_facts(
            message_id=message_id,
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
        ):
            return AdminBootstrapResult(False, "invalid_request_context")

        requested_target = (
            requested_target.strip()
            if isinstance(requested_target, str)
            else ""
        )
        if requested_target and not is_canonical_sender_id(requested_target):
            return AdminBootstrapResult(
                False,
                "invalid_target",
                admin_id=sender_id,
                target_id=requested_target,
            )

        with self._global_lock:
            now = time.time()
            last = self._last_attempt.get(sender_id, 0.0)
            if now - last < _RATE_LIMIT_SECONDS:
                return AdminBootstrapResult(
                    False,
                    "rate_limited",
                    admin_id=sender_id,
                )
            self._last_attempt[sender_id] = now

            settings = self._settings_getter()
            provider = self._provider_for(settings)
            base_policy = provider.current
            outcome: dict[str, object] = {}

            def mutate(current: Mapping[str, str]) -> dict[str, str]:
                values = current
                current_admins = self._ids_from_locked_state(
                    values,
                    "ADMIN_USER_IDS",
                    base_policy.admin_ids,
                )
                is_bootstrap = not current_admins
                if is_bootstrap:
                    if (
                        base_policy.admin_bootstrap_scope == "p2p_only"
                        and chat_type != "p2p"
                    ):
                        raise _EnrollmentRejected("bootstrap_requires_p2p")
                    target_id = sender_id
                else:
                    if sender_id not in current_admins:
                        raise _EnrollmentRejected("not_admin")
                    target_id = requested_target or sender_id

                if not is_canonical_sender_id(target_id):
                    raise _EnrollmentRejected(
                        "invalid_target",
                        target_id=target_id,
                    )

                updates = {"ADMIN_USER_IDS": target_id}
                if is_bootstrap:
                    allowed_users = self._ids_from_locked_state(
                        values,
                        "ALLOWED_USER_IDS",
                        base_policy.allowed_user_ids,
                    ) | {sender_id}
                    allowed_chats = self._ids_from_locked_state(
                        values,
                        "ALLOWED_CHAT_IDS",
                        base_policy.allowed_chat_ids,
                    ) | {chat_id}
                    updates.update(
                        {
                            "ALLOWED_USER_IDS": self._serialize_ids(
                                allowed_users
                            ),
                            "ALLOWED_CHAT_IDS": self._serialize_ids(
                                allowed_chats
                            ),
                        }
                    )
                outcome["is_bootstrap"] = is_bootstrap
                outcome["target_id"] = target_id
                return updates

            try:
                failure_code = self._persist_then_publish(
                    settings=settings,
                    provider=provider,
                    base_policy=base_policy,
                    mutator=mutate,
                    committed_keys=(
                        "ADMIN_USER_IDS",
                        "ALLOWED_USER_IDS",
                        "ALLOWED_CHAT_IDS",
                    ),
                )
            except _EnrollmentRejected as exc:
                return AdminBootstrapResult(
                    False,
                    exc.code,
                    admin_id=sender_id,
                    target_id=exc.target_id,
                )

            target_id = str(outcome.get("target_id", ""))
            if failure_code:
                return AdminBootstrapResult(
                    False,
                    failure_code,
                    admin_id=sender_id,
                    target_id=target_id,
                )

            result = AdminBootstrapResult(
                True,
                "bootstrap" if outcome.get("is_bootstrap") else "updated",
                admin_id=sender_id,
                target_id=target_id,
            )
            audit_logger.info(
                "ADMIN_CHANGE: sender_hash=%s target_hash=%s code=%s",
                self._hash_identifier(sender_id),
                self._hash_identifier(target_id),
                result.code,
            )
            return result

    def allow_current_chat(
        self,
        sender_id: str,
        chat_id: str,
        *,
        chat_type: str,
        message_id: str = "",
    ) -> AdminBootstrapResult:
        """Enroll the group where a configured admin ran the command."""

        if not are_canonical_ingress_facts(
            message_id=message_id,
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=chat_type,
        ):
            return AdminBootstrapResult(False, "invalid_request_context")
        if chat_type != "group":
            return AdminBootstrapResult(
                False,
                "access_requires_group",
                admin_id=sender_id,
                target_id=chat_id,
            )

        with self._global_lock:
            settings = self._settings_getter()
            provider = self._provider_for(settings)
            base_policy = provider.current
            outcome = {"already_enrolled": False}

            def mutate(current: Mapping[str, str]) -> dict[str, str]:
                current_admins = self._ids_from_locked_state(
                    current,
                    "ADMIN_USER_IDS",
                    base_policy.admin_ids,
                )
                if sender_id not in current_admins:
                    raise _EnrollmentRejected("not_admin")
                current_chats = self._ids_from_locked_state(
                    current,
                    "ALLOWED_CHAT_IDS",
                    base_policy.allowed_chat_ids,
                )
                if chat_id in current_chats:
                    outcome["already_enrolled"] = True
                    return {}
                return {
                    "ALLOWED_CHAT_IDS": self._serialize_ids(
                        current_chats | {chat_id}
                    )
                }

            try:
                failure_code = self._persist_then_publish(
                    settings=settings,
                    provider=provider,
                    base_policy=base_policy,
                    mutator=mutate,
                    committed_keys=("ALLOWED_CHAT_IDS",),
                )
            except _EnrollmentRejected as exc:
                return AdminBootstrapResult(
                    False,
                    exc.code,
                    admin_id=sender_id,
                    target_id=chat_id,
                )

            if failure_code:
                return AdminBootstrapResult(
                    False,
                    failure_code,
                    admin_id=sender_id,
                    target_id=chat_id,
                )

            audit_logger.info(
                "ACCESS_CHAT_ENROLMENT: sender_hash=%s chat_hash=%s",
                self._hash_identifier(sender_id),
                self._hash_identifier(chat_id),
            )
            return AdminBootstrapResult(
                True,
                (
                    "chat_already_enrolled"
                    if outcome["already_enrolled"]
                    else "chat_enrolled"
                ),
                admin_id=sender_id,
                target_id=chat_id,
            )

    @staticmethod
    def _ids_from_locked_state(
        current: Mapping[str, str],
        key: str,
        fallback: frozenset[str],
    ) -> frozenset[str]:
        if key in current:
            return normalize_access_ids(current[key])
        return fallback

    @staticmethod
    def _serialize_ids(values: frozenset[str] | set[str]) -> str:
        return ",".join(sorted(values))

    @staticmethod
    def _hash_identifier(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _provider_for(self, settings: object) -> IngressAccessPolicyProvider:
        provider = self._policy_provider
        if provider is None:
            provider = IngressAccessPolicyProvider(
                build_ingress_access_policy(settings)
            )
            self._policy_provider = provider
        return provider

    def _persist_then_publish(
        self,
        *,
        settings: object,
        provider: IngressAccessPolicyProvider,
        base_policy: IngressAccessPolicy,
        mutator: Callable,
        committed_keys: tuple[str, ...],
    ) -> str:
        try:
            committed = self._env_store.update_with(mutator)
        except _EnrollmentRejected:
            raise
        except EnvPreReplaceError:
            logger.error(
                "ACCESS_ENV_PRE_REPLACE_FAILED keys=%s",
                ",".join(sorted(committed_keys)),
                exc_info=True,
            )
            return "persistence_failed"
        except EnvCommitUncertainError as exc:
            provider.record_blocking_finding(
                _COMMIT_UNCERTAIN_FINDING,
                "dotenv replacement completed but directory durability is uncertain",
            )
            logger.critical(
                "ACCESS_ENV_COMMIT_UNCERTAIN keys=%s reconciled=%s",
                ",".join(sorted(committed_keys)),
                exc.snapshot is not None,
                exc_info=True,
            )
            if exc.snapshot is not None:
                refresh_failure = self._publish_committed_snapshot(
                    settings=settings,
                    provider=provider,
                    base_policy=base_policy,
                    committed=exc.snapshot,
                    clear_commit_uncertain=False,
                )
                if refresh_failure:
                    return "commit_uncertain_refresh_failed"
                return "commit_uncertain"
            return "commit_uncertain_unreconciled"
        except EnvPostCommitCleanupError as exc:
            provider.record_blocking_finding(
                _CLEANUP_FAILED_FINDING,
                "dotenv commit is durable but transaction resource cleanup failed",
            )
            logger.critical(
                "ACCESS_ENV_LOCK_CLEANUP_FAILED keys=%s",
                ",".join(sorted(committed_keys)),
                exc_info=True,
            )
            refresh_failure = self._publish_committed_snapshot(
                settings=settings,
                provider=provider,
                base_policy=base_policy,
                committed=exc.snapshot,
                clear_commit_uncertain=False,
            )
            if refresh_failure:
                return "commit_cleanup_refresh_failed"
            return "commit_cleanup_failed"
        except OSError:
            provider.record_blocking_finding(
                _COMMIT_UNCERTAIN_FINDING,
                "dotenv store returned an unclassified I/O failure",
            )
            logger.critical(
                "ACCESS_ENV_COMMIT_STATE_UNKNOWN keys=%s",
                ",".join(sorted(committed_keys)),
                exc_info=True,
            )
            try:
                reconciled = self._env_store.read_snapshot()
            except Exception:
                logger.critical(
                    "ACCESS_ENV_RECONCILIATION_FAILED",
                    exc_info=True,
                )
                return "commit_uncertain_unreconciled"
            if not isinstance(reconciled, EnvFileSnapshot):
                return "commit_uncertain_unreconciled"
            refresh_failure = self._publish_committed_snapshot(
                settings=settings,
                provider=provider,
                base_policy=base_policy,
                committed=reconciled,
                clear_commit_uncertain=False,
            )
            if refresh_failure:
                return "commit_uncertain_refresh_failed"
            return "commit_uncertain"

        return self._publish_committed_snapshot(
            settings=settings,
            provider=provider,
            base_policy=base_policy,
            committed=committed,
            clear_commit_uncertain=True,
        )

    def _publish_committed_snapshot(
        self,
        *,
        settings: object,
        provider: IngressAccessPolicyProvider,
        base_policy: IngressAccessPolicy,
        committed: EnvFileSnapshot,
        clear_commit_uncertain: bool,
    ) -> str:
        try:
            replacement_policy = self._policy_from_snapshot(
                base_policy,
                committed,
            )
        except Exception:
            provider.record_blocking_finding(
                _REFRESH_FAILED_FINDING,
                "committed ingress enrollment could not be published in-process",
            )
            logger.critical(
                "ACCESS_POLICY_REFRESH_BLOCKED",
                exc_info=True,
            )
            return "policy_refresh_failed"

        try:
            previous_settings = self._publish_settings_snapshot(
                settings,
                replacement_policy,
            )
        except Exception:
            provider.record_blocking_finding(
                _SETTINGS_MIRROR_FAILED_FINDING,
                "committed ingress enrollment could not be mirrored to settings",
            )
            logger.critical(
                "ACCESS_SETTINGS_MIRROR_BLOCKED",
                exc_info=True,
            )
            return "settings_mirror_failed"

        try:
            provider.swap(replacement_policy)
        except Exception:
            settings_restored = self._restore_settings_snapshot(
                settings,
                previous_settings,
            )
            if not settings_restored:
                provider.record_blocking_finding(
                    _SETTINGS_MIRROR_FAILED_FINDING,
                    "settings mirror rollback failed after policy publication error",
                )
            provider.record_blocking_finding(
                _REFRESH_FAILED_FINDING,
                "committed ingress enrollment could not be published in-process",
            )
            logger.critical(
                "ACCESS_POLICY_REFRESH_BLOCKED",
                exc_info=True,
            )
            return "policy_refresh_failed"

        provider.clear_blocking_finding(_REFRESH_FAILED_FINDING)
        provider.clear_blocking_finding(_SETTINGS_MIRROR_FAILED_FINDING)
        if clear_commit_uncertain:
            provider.clear_blocking_finding(_COMMIT_UNCERTAIN_FINDING)
            provider.clear_blocking_finding(_CLEANUP_FAILED_FINDING)
        return ""

    @staticmethod
    def _policy_from_snapshot(
        base_policy: IngressAccessPolicy,
        snapshot: EnvFileSnapshot,
    ) -> IngressAccessPolicy:
        values = snapshot.values

        def ids(key: str, fallback: frozenset[str]) -> frozenset[str]:
            if key not in values:
                return fallback
            return normalize_access_ids(values[key])

        return replace(
            base_policy,
            admin_ids=ids("ADMIN_USER_IDS", base_policy.admin_ids),
            allowed_user_ids=ids(
                "ALLOWED_USER_IDS",
                base_policy.allowed_user_ids,
            ),
            allowed_chat_ids=ids(
                "ALLOWED_CHAT_IDS",
                base_policy.allowed_chat_ids,
            ),
        )

    @classmethod
    def _publish_settings_snapshot(
        cls,
        settings: object,
        policy: IngressAccessPolicy,
    ) -> tuple[tuple[str, bool, object], ...]:
        previous: list[tuple[str, bool, object]] = []
        for field_name, value in (
            ("admin_user_ids", policy.admin_ids),
            ("allowed_user_ids", policy.allowed_user_ids),
            ("allowed_chat_ids", policy.allowed_chat_ids),
        ):
            if not hasattr(settings, field_name) and field_name != "admin_user_ids":
                continue
            existed = hasattr(settings, field_name)
            old_value = getattr(settings, field_name, None)
            try:
                object.__setattr__(settings, field_name, value)
            except Exception:
                cls._restore_settings_snapshot(settings, tuple(previous))
                raise
            previous.append((field_name, existed, old_value))
        return tuple(previous)

    @staticmethod
    def _restore_settings_snapshot(
        settings: object,
        previous: tuple[tuple[str, bool, object], ...],
    ) -> bool:
        restored = True
        for field_name, existed, old_value in reversed(previous):
            try:
                if existed:
                    object.__setattr__(settings, field_name, old_value)
                else:
                    object.__delattr__(settings, field_name)
            except Exception:
                restored = False
                logger.critical(
                    "ACCESS_SETTINGS_MIRROR_ROLLBACK_FAILED field=%s",
                    field_name,
                    exc_info=True,
                )
        return restored
