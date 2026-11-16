"""Employee runtime bridge, targeted handoff, and managed-group helpers for FeishuWSClient."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..autonomous.runtime.employee_handoff import _EmployeeTargetedCommand
    from ..trust.models import EffectiveTrust
    from .ws_client import (
        P2ImMessageReceiveV1,
        _MentionedGroupCommand,
        _MessageIngressReservation,
    )

logger = logging.getLogger(__name__)


def _ws_client_module():
    """Return the ws_client module for runtime-patchable symbols."""
    from . import ws_client as _ws_mod

    return _ws_mod


class WSEmployeeBridgeMixin:
    """Mixin providing employee runtime recovery, handoff, and group reconciliation."""

    def _recover_employee_runtime_after_handler_binding(
        self,
        *,
        close_on_failure: bool = True,
    ) -> object | None:
        """Start durable recovery only after the main-Bot reply path exists."""

        runtime = self._employee_department_runtime
        if runtime is None:
            return None
        try:
            if "coco" not in self._handler_ctx.handlers:
                raise RuntimeError("main Bot reply transport is not bound")
            return runtime.recover()
        except Exception:
            if close_on_failure:
                self._close_employee_runtime_after_initialization_failure()
            raise

    def _run_employee_runtime_recovery(self) -> None:
        """Recover Employee state after main WS readiness without weakening admission."""

        try:
            for recovery_attempt in range(2):
                try:
                    recovery_summary = (
                        self._recover_employee_runtime_after_handler_binding(
                            close_on_failure=False
                        )
                    )
                    break
                except Exception:
                    if recovery_attempt:
                        raise
                    logger.warning(
                        "Employee runtime background recovery failed once; "
                        "retrying once",
                        exc_info=True,
                    )
        except Exception as exc:
            self._employee_runtime_recovery_error = exc
            runtime = self._employee_department_runtime
            fail_recovery = getattr(runtime, "fail_recovery", None)
            if callable(fail_recovery):
                try:
                    fail_recovery("background_recovery")
                except Exception:
                    logger.exception(
                        "Employee runtime failed to close admission after recovery error"
                    )
            logger.exception(
                "Employee runtime background recovery failed; "
                "main Bot remains available and Employee admission stays closed"
            )
            return
        logger.info(
            "Employee runtime background recovery complete "
            "eligible=%d recovered=%d skipped=%d failed=%d",
            int(getattr(recovery_summary, "eligible", 0) or 0),
            int(getattr(recovery_summary, "recovered", 0) or 0),
            int(getattr(recovery_summary, "skipped", 0) or 0),
            int(getattr(recovery_summary, "failed", 0) or 0),
        )
        self._try_publish_restart_readiness()

    def _start_employee_runtime_recovery(self) -> None:
        """Start at most one post-connection Employee recovery worker."""

        if self._employee_department_runtime is None:
            return
        with self._employee_runtime_recovery_lock:
            if self._closed or self._employee_runtime_recovery_started:
                return
            self._employee_runtime_recovery_started = True
            thread = _ws_client_module().threading.Thread(
                target=self._run_employee_runtime_recovery,
                name="employee-runtime-recovery",
                daemon=True,
            )
            self._employee_runtime_recovery_thread = thread
            thread.start()

    def _wait_for_employee_runtime_recovery(self, timeout: float) -> bool:
        """Wait for background recovery before closing its durable resources."""

        lock = getattr(self, "_employee_runtime_recovery_lock", None)
        if lock is None:
            thread = getattr(self, "_employee_runtime_recovery_thread", None)
        else:
            with lock:
                thread = getattr(self, "_employee_runtime_recovery_thread", None)
        if thread is None or thread is _ws_client_module().threading.current_thread():
            return True
        if not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _close_employee_runtime_after_initialization_failure(self) -> None:
        """Close a composed runtime once without masking the init failure."""

        runtime = getattr(self, "_employee_department_runtime", None)
        if runtime is None or getattr(
            self,
            "_employee_runtime_init_cleanup_done",
            False,
        ):
            return
        self._employee_runtime_init_cleanup_done = True
        try:
            runtime.close()
        except BaseException:
            logger.error(
                "Employee Department cleanup failed after initialization error",
                exc_info=True,
            )

    def _resolve_ready_employee_target(
        self,
        candidate: _MentionedGroupCommand,
    ) -> _EmployeeTargetedCommand | None:
        """Freeze one READY employee authority on a scheduler worker."""

        from ..autonomous.provisioning.composition import (
            EmployeeTargetResolutionUnknownError,
        )

        runtime = getattr(self, "_employee_department_runtime", None)
        resolve_target = getattr(runtime, "resolve_ready_employee_bot_target", None)
        if not callable(resolve_target):
            raise EmployeeTargetResolutionUnknownError(
                "employee target resolver is unavailable"
            )
        try:
            target = resolve_target(
                tenant_key=candidate.tenant_key,
                chat_id=candidate.chat_id,
                bot_open_id=candidate.bot_open_id,
            )
        except EmployeeTargetResolutionUnknownError:
            logger.warning("employee Bot identity lookup failed closed", exc_info=True)
            raise
        except (OSError, TimeoutError) as exc:
            logger.warning("employee Bot identity lookup failed closed", exc_info=True)
            raise EmployeeTargetResolutionUnknownError(
                "employee target lookup dependency is unavailable"
            ) from exc
        if target is None:
            return None
        agent_id = getattr(target, "agent_id", None)
        bot_principal_id = getattr(target, "bot_principal_id", None)
        app_id = getattr(target, "app_id", None)
        channel_generation = getattr(target, "channel_generation", None)
        connection_id = getattr(target, "connection_id", None)
        if (
            getattr(target, "tenant_key", None) != candidate.tenant_key
            or getattr(target, "chat_id", None) != candidate.chat_id
            or getattr(target, "bot_open_id", None) != candidate.bot_open_id
            or not isinstance(agent_id, str)
            or not agent_id.startswith("agt_")
            or not isinstance(bot_principal_id, str)
            or not bot_principal_id.startswith("bot_")
            or not isinstance(app_id, str)
            or not app_id.startswith("cli_")
            or type(channel_generation) is not int
            or channel_generation <= 0
            or not isinstance(connection_id, str)
            or not connection_id.startswith("conn_")
        ):
            raise EmployeeTargetResolutionUnknownError(
                "employee target structure is invalid"
            )
        return _ws_client_module()._EmployeeTargetedCommand(
            tenant_key=candidate.tenant_key,
            chat_id=candidate.chat_id,
            message_id=candidate.message_id,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            bot_open_id=candidate.bot_open_id,
            channel_generation=channel_generation,
            connection_id=connection_id,
        )

    def _employee_handoff_timeout(self) -> float:
        """Wait up to timeout for employee ingress durability confirmation."""
        configured = getattr(
            self.settings,
            "autonomous_employee_ingress_ack_timeout_seconds",
            1.5,
        )
        if (
            isinstance(configured, bool)
            or not isinstance(configured, (int, float))
            or configured <= 0
        ):
            configured = 1.5
        return min(
            _ws_client_module()._EMPLOYEE_HANDOFF_MAX_SECONDS,
            float(configured) + _ws_client_module()._EMPLOYEE_HANDOFF_GRACE_SECONDS,
        )

    def _bind_message_ingress_reservation(
        self,
        handle: object,
        reservation: _MessageIngressReservation,
        *,
        tenant_key: str,
        chat_id: str,
    ) -> None:
        """Durably report an exact task that terminates before its callback starts."""

        if not reservation.owns():
            return

        add_done_callback = getattr(handle, "add_done_callback", None)
        if not callable(add_done_callback):
            reservation.release_if_unstarted()
            raise TypeError("scheduler task handle lacks completion callbacks")

        def report_if_unstarted(_event: object) -> None:
            if not reservation.claim_unstarted_terminal():
                return
            anchored = False
            try:
                runtime = getattr(self, "_employee_department_runtime", None)
                queue_warning = getattr(runtime, "queue_main_bot_warning", None)
                if callable(queue_warning):
                    anchored = queue_warning(
                        tenant_key=tenant_key,
                        chat_id=chat_id,
                        message_id=reservation.message_id,
                        text=_ws_client_module().UI_TEXT["ws_message_prestart_terminal"],
                    ) is True
                else:
                    logger.error(
                        "pre-start terminal durable warning queue is unavailable"
                    )
            except Exception:
                logger.error(
                    "pre-start terminal durable warning prepare failed",
                    exc_info=True,
                )
            finally:
                try:
                    if anchored:
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
        employee_candidate: _MentionedGroupCommand | None,
    ) -> None:
        """Execute an admitted reserved message callback safely with error reporting."""
        if not reservation.mark_started():
            return
        self._process_message_async(
            data,
            task_ctx=task_ctx,
            shell_fast_tracked=shell_fast_tracked,
            effective_trust=effective_trust,
            employee_candidate=employee_candidate,
            message_reservation_id=reservation.message_id,
            message_reservation_owner=reservation.owner,
        )

    def _employee_target_handoff_confirmed(
        self,
        target: _EmployeeTargetedCommand,
    ) -> bool | None:
        """Return True once employee ingress reports durable acceptance."""
        from ..autonomous.provisioning.composition import (
            EmployeeMessageHandoffUnknownError,
        )

        runtime = getattr(self, "_employee_department_runtime", None)
        wait_for_handoff = getattr(
            runtime,
            "wait_for_employee_message_handoff",
            None,
        )
        if not callable(wait_for_handoff):
            logger.warning(
                "employee targeted command handoff proof API unavailable"
            )
            return None
        try:
            result = wait_for_handoff(
                tenant_key=target.tenant_key,
                agent_id=target.agent_id,
                bot_principal_id=target.bot_principal_id,
                app_id=target.app_id,
                channel_generation=target.channel_generation,
                connection_id=target.connection_id,
                chat_id=target.chat_id,
                message_id=target.message_id,
                timeout=self._employee_handoff_timeout(),
            )
        except (EmployeeMessageHandoffUnknownError, OSError, TimeoutError):
            logger.warning(
                "employee targeted command acceptance proof unavailable",
                exc_info=True,
            )
            return None
        if result is True:
            return True
        if result is False:
            return False
        logger.warning(
            "employee targeted command handoff proof returned invalid result"
        )
        return None

    def _complete_employee_target_handoff(
        self,
        target: _EmployeeTargetedCommand,
    ) -> bool:
        """Return whether this origin must remain fenced after handoff."""

        confirmed = self._employee_target_handoff_confirmed(target)
        if confirmed is True:
            return True
        if confirmed is False:
            return self._reply_employee_handoff_unconfirmed(target)
        # UNKNOWN may mean the Employee already executed.  Warning persistence
        # failure therefore cannot authorize a competing main-Bot retry.
        self._reply_employee_handoff_unknown(target)
        return True

    def _reply_employee_handoff_unknown(
        self,
        target: _EmployeeTargetedCommand | _MentionedGroupCommand,
    ) -> bool:
        """Notify sender that the targeted @employee could not be verified."""
        return self._queue_employee_handoff_warning(
            target,
            _ws_client_module().UI_TEXT["ws_employee_handoff_unknown"],
            label="employee targeted command handoff unknown",
        )

    def _reply_employee_handoff_unconfirmed(
        self,
        target: _EmployeeTargetedCommand | _MentionedGroupCommand,
    ) -> bool:
        """Notify sender when employee handoff confirmation exceeded its wait budget."""
        return self._queue_employee_handoff_warning(
            target,
            _ws_client_module().UI_TEXT["ws_employee_handoff_unconfirmed"],
            label="employee targeted command handoff warning",
        )

    def _queue_employee_handoff_warning(
        self,
        target: _EmployeeTargetedCommand | _MentionedGroupCommand,
        text: str,
        *,
        label: str,
    ) -> bool:
        """Prepare one warning without performing transport I/O on this task."""

        from ..autonomous.provisioning.composition import (
            MainBotWarningPreparationError,
        )

        runtime = getattr(self, "_employee_department_runtime", None)
        queue_warning = getattr(runtime, "queue_main_bot_warning", None)
        if not callable(queue_warning):
            logger.error("%s durable queue is unavailable", label)
            return False
        try:
            anchored = queue_warning(
                tenant_key=target.tenant_key,
                chat_id=target.chat_id,
                message_id=target.message_id,
                text=text,
            )
        except MainBotWarningPreparationError:
            logger.error("%s durable prepare failed", label, exc_info=True)
            return False
        if anchored is not True:
            logger.error("%s durable queue returned invalid result", label)
            return False
        return True

    def _reply_text_with_bounded_retry(
        self,
        message_id: str,
        text: str,
        *,
        label: str,
    ) -> bool:
        """Deliver one user-visible terminal warning with bounded recovery."""

        idempotency_key = _ws_client_module()._employee_warning_uuid(message_id, label)
        for attempt, delay in enumerate(_ws_client_module()._EMPLOYEE_WARNING_RETRY_DELAYS, start=1):
            if delay:
                _ws_client_module().time.sleep(delay)
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
            if attempt < len(_ws_client_module()._EMPLOYEE_WARNING_RETRY_DELAYS):
                logger.debug("%s delivery retry %d", label, attempt)
                continue
            logger.warning("%s delivery failed", label)
        return False

    def _schedule_employee_handoff_warning(
        self,
        candidate: _MentionedGroupCommand,
        *,
        reservation: _MessageIngressReservation,
        request_id: str,
        sender_id: str,
        sender_union_id: str,
        tenant_key: str,
    ) -> bool:
        """Transfer one owned ingress reservation to the system warning lane."""

        spec = _ws_client_module().TaskSpec(
            chat_id=candidate.chat_id,
            name="employee_handoff_warning",
            task_type="employee_handoff_warning",
            message_id=candidate.message_id,
            origin_message_id=candidate.message_id,
            request_id=request_id,
            priority=_ws_client_module().TaskPriority.HIGH,
            is_system_command=True,
            is_p2p=False,
            sender_id=sender_id,
            sender_union_id=sender_union_id,
            tenant_key=tenant_key,
        )

        def warn(_ctx) -> None:
            if not reservation.mark_started():
                return
            if not reservation.owns():
                return
            try:
                main_bot_open_id = (
                    self._main_bot_open_id or self._sync_main_bot_identity()
                )
                if not main_bot_open_id:
                    delivered = self._reply_text_with_bounded_retry(
                        candidate.message_id,
                        _ws_client_module().UI_TEXT["ws_main_bot_identity_unavailable"],
                        label="main Bot identity warning",
                    )
                elif candidate.bot_open_id == main_bot_open_id:
                    delivered = self._reply_text_with_bounded_retry(
                        candidate.message_id,
                        _ws_client_module().UI_TEXT["ws_backpressure_generic"],
                        label="main Bot backpressure warning",
                    )
                else:
                    from ..autonomous.provisioning.composition import (
                        EmployeeTargetResolutionUnknownError,
                    )

                    try:
                        target = self._resolve_ready_employee_target(candidate)
                    except EmployeeTargetResolutionUnknownError:
                        try:
                            self._reply_employee_handoff_unknown(candidate)
                        finally:
                            # UNKNOWN may conceal completed Employee execution.
                            reservation.commit()
                        return
                    if target is not None:
                        confirmed = self._employee_target_handoff_confirmed(target)
                        if confirmed is False:
                            if self._reply_employee_handoff_unconfirmed(candidate):
                                reservation.commit()
                        elif confirmed is None:
                            try:
                                self._reply_employee_handoff_unknown(candidate)
                            finally:
                                # UNKNOWN may conceal completed Employee execution.
                                reservation.commit()
                        else:
                            reservation.commit()
                        return
                    delivered = self._reply_employee_handoff_unconfirmed(candidate)
                if delivered:
                    reservation.commit()
            finally:
                reservation.release()

        try:
            handle = self._scheduler.submit(spec, warn)
        except (
            _ws_client_module().RateLimitExceededException,
            _ws_client_module().CircuitBreakerOpenException,
            _ws_client_module().TaskQueueFullError,
            RuntimeError,
            OSError,
            TimeoutError,
            TypeError,
            ValueError,
        ):
            reservation.release()
            logger.warning(
                "employee handoff warning admission failed closed",
                exc_info=True,
            )
            return False
        self._bind_message_ingress_reservation(
            handle,
            reservation,
            tenant_key=tenant_key,
            chat_id=candidate.chat_id,
        )
        return True

    def _reply_employee_hire_status(self, state: object, status: str) -> object | None:
        """Render and send an employee hire lifecycle status update to chat."""
        text = _ws_client_module()._employee_hire_status_text(
            str(getattr(state, "employee_name", "")),
            status,
        )
        if text is None:
            return None
        intent_id = getattr(state, "intent_id", "")
        if not isinstance(intent_id, str) or not intent_id:
            return None
        return self._reply_employee_hire_message(
            state,
            text,
            idempotency_key=_ws_client_module()._employee_hire_status_uuid(intent_id, status),
        )

    def _reply_employee_hire_message(
        self,
        state: object,
        text: str,
        *,
        idempotency_key: str | None = None,
    ) -> object | None:
        """Restore durable requester scope before a recovered hire reply."""

        if not self._restore_employee_hire_origin(state):
            logger.warning("employee hire notification recipient scope is unavailable")
            return None
        message_id = getattr(state, "message_id", "")
        tenant_key = getattr(state, "tenant_key", "")
        previous_tenant_key = _ws_client_module().get_current_tenant_key()
        _ws_client_module().set_current_tenant_key(tenant_key or None)
        try:
            return self._handler_ctx.handlers["coco"].reply_text(
                message_id,
                text,
                idempotency_key=idempotency_key,
            )
        finally:
            _ws_client_module().set_current_tenant_key(previous_tenant_key)

    def _reply_employee_team_message(
        self,
        message_id: str,
        chat_id: str,
        text: str,
        *,
        tenant_key: str,
        requester_principal_id: str,
        idempotency_key: str | None = None,
    ) -> object | None:
        """Restore durable Team Run requester scope before a recovered reply."""

        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                message_id,
                chat_id,
                tenant_key,
                requester_principal_id,
            )
        ):
            logger.warning("employee team notification recipient scope is unavailable")
            raise RuntimeError("employee team notification recipient scope is unavailable")
        if not self._restore_durable_message_origin(
            message_id=message_id,
            chat_id=chat_id,
            sender_id=requester_principal_id,
            tenant_key=tenant_key,
        ):
            logger.warning("employee team notification recipient scope is unavailable")
            raise RuntimeError("employee team notification recipient scope is unavailable")
        previous_tenant_key = _ws_client_module().get_current_tenant_key()
        _ws_client_module().set_current_tenant_key(tenant_key)
        try:
            reply_id = self._handler_ctx.handlers["coco"].reply_text(
                message_id,
                text,
                idempotency_key=idempotency_key,
            )
            if not reply_id:
                raise RuntimeError("employee team notification delivery failed")
            return reply_id
        finally:
            _ws_client_module().set_current_tenant_key(previous_tenant_key)

    def _restore_employee_hire_origin(self, state: object) -> bool:
        """Link an employee hire notification message back to its origin request."""
        message_id = getattr(state, "message_id", "")
        chat_id = getattr(state, "chat_id", "")
        sender_id = getattr(state, "requester_principal_id", "")
        tenant_key = getattr(state, "tenant_key", "")
        if not all(
            isinstance(value, str) and bool(value)
            for value in (message_id, chat_id, sender_id)
        ) or not isinstance(tenant_key, str):
            return False
        return self._restore_durable_message_origin(
            message_id=message_id,
            chat_id=chat_id,
            sender_id=sender_id,
            tenant_key=tenant_key,
        )

    def _restore_durable_message_origin(
        self,
        *,
        message_id: str,
        chat_id: str,
        sender_id: str,
        tenant_key: str,
    ) -> bool:
        """Rebuild trusted in-memory provenance from anchored durable facts."""

        try:
            origin = self._message_linker.query(message_id)
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return False
        if origin is not None:
            chat_type = origin.get("chat_type")
            if chat_type not in {"p2p", "group", "topic_group"}:
                return False
        else:
            chat_mode = self._get_chat_mode(chat_id)
            if chat_mode is None:
                return False
            chat_type = "topic_group" if chat_mode == "topic" else chat_mode
        try:
            return (
                self._message_linker.register_trusted_origin_with_tenant(
                    message_id,
                    chat_id=chat_id,
                    sender_id=sender_id,
                    chat_type=chat_type,
                    tenant_key=tenant_key,
                )
                is True
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
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

        if not _ws_client_module()._visible_employee_runtime_requires_outbound_audit(
            getattr(self, "settings", None)
        ):
            return True, ""

        audit = getattr(handler_ctx, "main_bot_outbound_audit", None)
        audit_failure = getattr(
            handler_ctx,
            "main_bot_outbound_audit_failure",
            None,
        )
        if audit is _ws_client_module()._unavailable_main_bot_outbound_audit or not callable(audit):
            return False, "main_bot_outbound_audit"
        if not callable(audit_failure):
            return False, "main_bot_outbound_audit_failure"
        return True, ""
