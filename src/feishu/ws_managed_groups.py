"""Managed-group reconciliation and ingress reservation helpers for FeishuWSClient."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..trust.models import EffectiveTrust
    from .ws_client import P2ImMessageReceiveV1, _MessageIngressReservation

logger = logging.getLogger(__name__)


def _ws_client_module():
    """Return the ws_client module for runtime-patchable symbols."""
    from . import ws_client as _ws_mod

    return _ws_mod


class WSManagedGroupsMixin:
    """Mixin providing Main Bot ingress reservations and managed-group reconciliation."""

    def _bind_message_ingress_reservation(
        self,
        handle: object,
        reservation: _MessageIngressReservation,
        *,
        tenant_key: str = "",
        chat_id: str = "",
    ) -> None:
        """Report an exact task that terminates before its callback starts."""
        del tenant_key, chat_id
        if not reservation.owns():
            return

        add_done_callback = getattr(handle, "add_done_callback", None)
        if not callable(add_done_callback):
            reservation.release_if_unstarted()
            raise TypeError("scheduler task handle lacks completion callbacks")

        def report_if_unstarted(_event: object) -> None:
            if not reservation.claim_unstarted_terminal():
                return
            delivered = False
            try:
                ws_mod = _ws_client_module()
                delivered = self._reply_text_with_bounded_retry(
                    reservation.message_id,
                    ws_mod.UI_TEXT["ws_message_prestart_terminal"],
                    label="pre-start terminal warning",
                )
            except Exception:
                logger.error(
                    "pre-start terminal warning delivery failed",
                    exc_info=True,
                )
            finally:
                try:
                    if delivered:
                        reservation.commit()
                    else:
                        # The scheduler terminal proves this callback never
                        # started, so platform redelivery cannot duplicate work.
                        reservation.release()
                except Exception:
                    logger.error(
                        "message ingress terminal disposition failed",
                        exc_info=True,
                    )

        try:
            add_done_callback(report_if_unstarted)
        except Exception:
            reservation.release_if_unstarted()
            raise

    def _run_reserved_message(
        self,
        reservation: _MessageIngressReservation,
        data: P2ImMessageReceiveV1,
        *,
        task_ctx: object,
        shell_fast_tracked: bool,
        effective_trust: EffectiveTrust | None,
    ) -> None:
        """Execute an admitted reserved message callback safely with error reporting."""
        if not reservation.mark_started():
            return
        self._process_message_async(
            data,
            task_ctx=task_ctx,
            shell_fast_tracked=shell_fast_tracked,
            effective_trust=effective_trust,
            message_reservation_id=reservation.message_id,
            message_reservation_owner=reservation.owner,
        )

    def _reply_text_with_bounded_retry(
        self,
        message_id: str,
        text: str,
        *,
        label: str,
    ) -> bool:
        """Deliver one user-visible terminal warning with bounded recovery."""

        ws_mod = _ws_client_module()
        idempotency_key = ws_mod._main_warning_uuid(message_id, label)
        for attempt, delay in enumerate(ws_mod._MAIN_WARNING_RETRY_DELAYS, start=1):
            if delay:
                ws_mod.time.sleep(delay)
            try:
                reply_id = self._handler_ctx.handlers["coco"].reply_text(
                    message_id,
                    text,
                    idempotency_key=idempotency_key,
                )
                if isinstance(reply_id, str) and reply_id:
                    return True
            except (RuntimeError, OSError, TimeoutError, TypeError, ValueError):
                logger.debug(
                    "%s delivery attempt %d raised",
                    label,
                    attempt,
                    exc_info=True,
                )
            if attempt < len(ws_mod._MAIN_WARNING_RETRY_DELAYS):
                logger.debug("%s delivery retry %d", label, attempt)
                continue
            logger.warning("%s delivery failed", label)
        return False

    def _reconcile_managed_groups_before_team_start(self) -> None:
        """Reconcile pending revokes and validated legacy Project candidates."""

        from ..project_chat.lark_chat_client import (
            LarkChatClient,
            ManagedChatValidation,
        )

        # Resolve an anchored pre/post-replace state before interpreting grants
        # or running any compensation against remote chats.
        self._managed_group_registry.reconcile_uncertain_commit()
        remote = LarkChatClient(api_client_factory=self._get_api_client)

        # Project binding sagas are authoritative lifecycle work, not legacy
        # migration candidates. Consume them before scanning bound projects so
        # OWNER_ADOPTED provenance cannot be rewritten as GHOSTAP_CREATED.
        from ..trust.registry import RegistryCommitUncertainError

        for saga in self._project_manager.pending_managed_chat_binding_sagas():
            project = self._project_manager.get_project_for_diagnostics(
                saga.project_id
            )
            active = self._managed_group_registry.active_record(saga.chat_id)
            grant = self._managed_group_registry.grant_for_chat(saga.chat_id)
            expected_origin = saga.expected_origin or (
                "owner_adopted"
                if saga.operation_id.startswith("adopt:")
                else "ghostap_created"
            )
            expected_owner = saga.expected_owner_id or self._managed_group_owner_id
            expected_bot = (
                saga.expected_receiving_bot_ref
                or self._managed_group_receiving_bot_ref
            )
            expected_root = saga.expected_root_ref or (
                project.root_path if project is not None else ""
            )
            active_matches = (
                project is not None
                and project.bound_chat_id == saga.chat_id
                and project.root_path == expected_root
                and active is not None
                and grant is not None
                and active.project_id == saga.project_id
                and active.canonical_root_ref == expected_root
                and active.origin.value == expected_origin
                and active.owner_id == expected_owner
                and active.receiving_bot_ref == expected_bot
                and grant.managed_group_id == saga.chat_id
                and grant.owner_id == expected_owner
                and grant.project_id == saga.project_id
                and grant.canonical_root_ref == expected_root
                and not grant.backend_binding_ids
                and not grant.connected_target_refs
            )
            if saga.expected is None:
                if active_matches:
                    self._project_manager.complete_exact_active_legacy_saga(
                        saga.operation_id
                    )
                else:
                    self._project_manager.mark_legacy_saga_resolution_required(
                        saga.operation_id
                    )
                continue
            if not self._project_manager.validate_managed_chat_binding_saga(
                saga.operation_id
            ):
                continue
            if active_matches:
                self._project_manager.complete_managed_chat_binding_saga(
                    saga.operation_id
                )
                continue
            if active is not None:
                self._project_manager.quarantine_bound_chat(saga.chat_id)
                continue

            binding = self._managed_group_registry.provision_binding(
                saga.operation_id
            )
            if (
                project is not None
                and binding is not None
                and binding.chat_id == saga.chat_id
                and binding.project_id == saga.project_id
                and binding.canonical_root_ref == expected_root
                and binding.origin.value == expected_origin
                and binding.owner_id == expected_owner
                and binding.receiving_bot_ref == expected_bot
            ):
                try:
                    self._managed_group_registry.activate(
                        provision_id=saga.operation_id,
                        chat_id=saga.chat_id,
                        project_id=saga.project_id,
                        canonical_root_ref=expected_root,
                    )
                except RegistryCommitUncertainError as exc:
                    if exc.committed:
                        continue
                    raise
                except Exception:
                    logger.exception(
                        "managed group binding saga activation failed operation=%s",
                        saga.operation_id,
                    )
                else:
                    self._project_manager.complete_managed_chat_binding_saga(
                        saga.operation_id
                    )
                    continue

            restored = self._project_manager.restore_managed_chat_binding_saga(
                saga.operation_id
            )
            if restored:
                try:
                    self._managed_group_registry.abandon_provision(
                        saga.operation_id
                    )
                except RegistryCommitUncertainError:
                    logger.exception(
                        "managed group binding saga cleanup uncertain operation=%s",
                        saga.operation_id,
                    )
            else:
                self._project_manager.quarantine_bound_chat(saga.chat_id)

        for chat_id in self._managed_group_registry.pending_revokes():
            result = remote.delete_chat(chat_id)
            if result is False:
                continue
            if not self._project_manager.revoke_managed_chat(chat_id):
                continue
            self._managed_group_registry.complete_revoke(chat_id)

        pending_saga_chat_ids = {
            saga.chat_id
            for saga in self._project_manager.pending_managed_chat_binding_sagas()
        }
        candidates_by_chat: dict[str, list[dict]] = {}
        for project in self._project_manager.get_all_projects(
            sort_by_recent=False,
            chat_id=None,
        ):
            candidate = project.managed_group_migration_candidate(
                owner_id=self._managed_group_owner_id,
                receiving_bot_ref=self._managed_group_receiving_bot_ref,
            )
            if candidate is None:
                continue
            chat_id = candidate["chat_id"]
            if chat_id in pending_saga_chat_ids:
                continue
            candidates_by_chat.setdefault(chat_id, []).append(candidate)

        for chat_id, candidates in sorted(candidates_by_chat.items()):
            if self._managed_group_registry.record(chat_id) is not None:
                continue
            distinct_bindings = {
                (candidate["project_id"], candidate["canonical_root_ref"])
                for candidate in candidates
            }
            if len(distinct_bindings) != 1:
                project_ids = ",".join(
                    sorted({candidate["project_id"] for candidate in candidates})
                )
                self._managed_group_registry.record_migration_disposition(
                    chat_id,
                    project_id=project_ids,
                    status="ambiguous",
                )
                logger.warning(
                    "managed group migration is ambiguous chat=%s projects=%s",
                    chat_id[:12],
                    project_ids,
                )
                continue
            candidate = candidates[0]
            validation = remote.validate_managed_chat(
                chat_id,
                self._managed_group_owner_id,
            )
            if validation is ManagedChatValidation.VALID:
                self._managed_group_registry.import_candidate(
                    candidate,
                    validator=lambda _facts: True,
                )
            else:
                existing = {
                    item[0]: (item[1], item[2])
                    for item in self._managed_group_registry.migration_dispositions()
                }
                disposition = (candidate["project_id"], validation.value)
                self._managed_group_registry.record_migration_disposition(
                    chat_id,
                    project_id=candidate["project_id"],
                    status=validation.value,
                )
                if existing.get(chat_id) != disposition:
                    logger.warning(
                        "managed group migration requires Owner review chat=%s disposition=%s",
                        chat_id[:12],
                        validation.value,
                    )

        for chat_id, project_id, status in (
            self._managed_group_registry.unreported_migration_dispositions()
        ):
            if status != "ambiguous":
                continue
            sent = remote.send_text_to_open_id(
                self._managed_group_owner_id,
                "受管群迁移需要人工处理："
                f"群 {chat_id}，项目 {project_id}，状态 {status}。"
                "请在 Owner 私聊中使用 /access migration-status 查看。",
            )
            if sent:
                self._managed_group_registry.mark_migration_reported(chat_id)

    def rotate_main_managed_group_bot(
        self,
        expected_bot_ref: str,
    ) -> tuple[int, int]:
        """CAS-rotate records for the explicitly identified former main Bot."""

        from ..project_chat.lark_chat_client import (
            LarkChatClient,
            ManagedChatValidation,
        )

        if not expected_bot_ref or expected_bot_ref == self._managed_group_receiving_bot_ref:
            return 0, 0
        remote = LarkChatClient(api_client_factory=self._get_api_client)
        rotated = 0
        rejected = 0
        for record in self._managed_group_registry.managed_groups():
            if (
                getattr(record.status, "value", "") != "active"
                or record.receiving_bot_ref != expected_bot_ref
            ):
                continue
            validation = remote.validate_managed_chat(
                record.chat_id,
                self._managed_group_owner_id,
            )
            if validation is not ManagedChatValidation.VALID:
                rejected += 1
                continue
            self._managed_group_registry.rotate_receiving_bot(
                chat_id=record.chat_id,
                expected_bot_ref=expected_bot_ref,
                new_bot_ref=self._managed_group_receiving_bot_ref,
            )
            rotated += 1
        return rotated, rejected

    def _main_bot_reply_chain_readiness(self) -> tuple[bool, str]:
        """Return whether the connected main Bot can emit an audited reply."""

        handler_ctx = getattr(self, "_handler_ctx", None)
        handlers = getattr(handler_ctx, "handlers", None)
        try:
            reply_handler_ready = handlers is not None and "coco" in handlers
        except Exception:
            reply_handler_ready = False
        if not reply_handler_ready:
            return False, "main_bot_reply_handler"
        return True, ""
