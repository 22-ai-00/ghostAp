"""Deterministic, fail-closed employee Thread context assembly."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import replace
from typing import Callable

from ..authorization import EmployeeAuthorizationScope
from .budget import apply_context_budget
from .models import (
    AssembledContext,
    ContextLayer,
    ContextLayerMetrics,
    ContextMessage,
    ContextUnavailableError,
    ContextUnavailableReason,
    EmployeeMessageScope,
    MessageRevision,
    ThreadContextConfig,
    ThreadWatermark,
)
from .source import EmployeeScopedMessageSource, MessagePage, ResolvedThread

_MAX_STABILITY_ATTEMPTS = 2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EmployeeThreadContext:
    """Build one immutable context snapshot through the current message."""

    def __init__(
        self,
        *,
        message_source: EmployeeScopedMessageSource,
        config: ThreadContextConfig | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = message_source
        self._config = config or ThreadContextConfig()
        self._monotonic = monotonic

    def assemble(
        self,
        *,
        l1_summary: str = "",
        l2_summary: str = "",
        system_prompt_token_reserve: int = 0,
        constraints_digest: str = "",
    ) -> AssembledContext:
        """Assemble a stable snapshot or raise a non-secret typed failure."""
        failure_reason: ContextUnavailableReason | None = None
        try:
            self._validate_trusted_inputs(
                l1_summary=l1_summary,
                l2_summary=l2_summary,
                reserve=system_prompt_token_reserve,
                constraints_digest=constraints_digest,
            )
            deadline = self._monotonic() + self._config.fetch_timeout_seconds
            scope = self._source.scope
            resolved = self._checked_call(self._source.resolve_thread, deadline)
            self._validate_binding(scope, resolved)
            stable_thread = self._stable_thread(scope, resolved, deadline)
            second_group: list[ContextMessage] = []
            if (
                scope.authorization_scope
                is EmployeeAuthorizationScope.MANAGED_GROUP
            ):
                first_group = self._fetch_group_window(
                    scope,
                    stable_thread,
                    deadline,
                )
                second_group = self._fetch_group_window(
                    scope,
                    stable_thread,
                    deadline,
                )
                if self._revision_digest(first_group) != self._revision_digest(
                    second_group
                ):
                    raise ContextUnavailableError(
                        ContextUnavailableReason.REVISION
                    )
            thread, group = self._reconcile_layers(
                scope,
                stable_thread,
                second_group,
            )
            watermark = self._watermark(scope, resolved, thread)
            return self._apply_budget(
                scope=scope,
                watermark=watermark,
                thread=thread,
                group=group,
                l1_summary=l1_summary,
                l2_summary=l2_summary,
                reserve=system_prompt_token_reserve,
                constraints_digest=constraints_digest,
            )
        except ContextUnavailableError as exc:
            failure_reason = exc.reason
        except Exception:
            failure_reason = ContextUnavailableReason.SOURCE
        # Construct the public error outside the handling block. ``from None``
        # only suppresses display; raising inside ``except`` would retain the
        # secret-bearing upstream exception in ``__context__``.
        raise ContextUnavailableError(failure_reason) from None

    @staticmethod
    def _validate_trusted_inputs(
        *,
        l1_summary: str,
        l2_summary: str,
        reserve: int,
        constraints_digest: str,
    ) -> None:
        if not isinstance(l1_summary, str) or not isinstance(l2_summary, str):
            raise ContextUnavailableError(ContextUnavailableReason.MEMORY)
        if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
            raise ContextUnavailableError(ContextUnavailableReason.BUDGET)
        if not isinstance(constraints_digest, str):
            raise ContextUnavailableError(ContextUnavailableReason.SCOPE)
        if reserve and not constraints_digest:
            raise ContextUnavailableError(ContextUnavailableReason.SCOPE)
        if constraints_digest and _SHA256_RE.fullmatch(constraints_digest) is None:
            raise ContextUnavailableError(ContextUnavailableReason.SCOPE)

    def _checked_call(self, operation: Callable[[], object], deadline: float):
        self._check_deadline(deadline)
        result = operation()
        self._check_deadline(deadline)
        return result

    def _check_deadline(self, deadline: float) -> None:
        if self._monotonic() >= deadline:
            raise ContextUnavailableError(ContextUnavailableReason.DEADLINE)

    @staticmethod
    def _validate_binding(
        scope: EmployeeMessageScope,
        resolved: ResolvedThread,
    ) -> None:
        plain_group_root = not resolved.feishu_thread_id
        if (
            resolved.thread_root_message_id != scope.thread_root_message_id
            or resolved.current_message_id != scope.current_message_id
            or (
                plain_group_root
                and scope.thread_root_message_id != scope.current_message_id
            )
            or (
                scope.feishu_thread_id
                and resolved.feishu_thread_id != scope.feishu_thread_id
            )
        ):
            raise ContextUnavailableError(
                ContextUnavailableReason.ROOT_THREAD_BINDING
            )

    def _stable_thread(
        self,
        scope: EmployeeMessageScope,
        resolved: ResolvedThread,
        deadline: float,
    ) -> list[ContextMessage]:
        saw_current = False
        for _ in range(_MAX_STABILITY_ATTEMPTS):
            first = self._thread_traversal(scope, resolved, deadline)
            second = self._thread_traversal(scope, resolved, deadline)
            saw_current = saw_current or first is not None or second is not None
            if first is None or second is None:
                continue
            if self._revision_digest(first) == self._revision_digest(second):
                return second
        reason = (
            ContextUnavailableReason.REVISION
            if saw_current
            else ContextUnavailableReason.CURRENT_MESSAGE
        )
        raise ContextUnavailableError(reason)

    def _thread_traversal(
        self,
        scope: EmployeeMessageScope,
        resolved: ResolvedThread,
        deadline: float,
    ) -> list[ContextMessage] | None:
        raw = self._collect_pages(
            lambda token: self._source.list_thread_messages(
                page_token=token,
                page_size=self._config.thread_page_size,
            ),
            deadline=deadline,
        )
        messages = self._deduplicate(raw)
        self._validate_thread_scope(scope, resolved, messages)
        current_indexes = [
            index
            for index, message in enumerate(messages)
            if message.message_id == scope.current_message_id
        ]
        if not current_indexes:
            return None
        if len(current_indexes) != 1:
            raise ContextUnavailableError(ContextUnavailableReason.REVISION)
        current = messages[current_indexes[0]]
        if current.deleted or current.is_system or current.msg_type == "system":
            raise ContextUnavailableError(ContextUnavailableReason.CURRENT_MESSAGE)
        if current.sender_tenant_key != scope.tenant_key:
            raise ContextUnavailableError(ContextUnavailableReason.SCOPE)
        if (
            current.sender_id != scope.source_requester_principal_id
            or current.sender_id_type != "open_id"
            or current.sender_type != "user"
        ):
            raise ContextUnavailableError(ContextUnavailableReason.SCOPE)
        prefix = messages[: current_indexes[0] + 1]
        if len(prefix) > self._config.max_thread_messages:
            raise ContextUnavailableError(ContextUnavailableReason.BUDGET)
        return [
            replace(message, is_current=message.message_id == scope.current_message_id)
            for message in prefix
        ]

    def _collect_pages(
        self,
        fetch: Callable[[str], MessagePage],
        *,
        deadline: float,
    ) -> list[ContextMessage]:
        messages: list[ContextMessage] = []
        token = ""
        seen_tokens: set[str] = set()
        for page_number in range(1, self._config.max_pages + 1):
            page = self._checked_call(lambda: fetch(token), deadline)
            if not isinstance(page, MessagePage):
                raise ContextUnavailableError(ContextUnavailableReason.SOURCE)
            messages.extend(page.messages)
            if not page.has_more:
                return messages
            next_token = page.page_token
            if not next_token or next_token in seen_tokens:
                raise ContextUnavailableError(ContextUnavailableReason.PAGINATION)
            seen_tokens.add(next_token)
            token = next_token
            if page_number == self._config.max_pages:
                raise ContextUnavailableError(ContextUnavailableReason.PAGINATION)
        raise ContextUnavailableError(ContextUnavailableReason.PAGINATION)

    @staticmethod
    def _validate_thread_scope(
        scope: EmployeeMessageScope,
        resolved: ResolvedThread,
        messages: list[ContextMessage],
    ) -> None:
        if not resolved.feishu_thread_id and (
            len(messages) != 1
            or messages[0].message_id != scope.current_message_id
            or scope.thread_root_message_id != scope.current_message_id
        ):
            raise ContextUnavailableError(
                ContextUnavailableReason.ROOT_THREAD_BINDING
            )
        for message in messages:
            if message.chat_id != scope.chat_id:
                raise ContextUnavailableError(ContextUnavailableReason.SCOPE)
            if message.thread_id != resolved.feishu_thread_id:
                raise ContextUnavailableError(
                    ContextUnavailableReason.ROOT_THREAD_BINDING
                )
            if message.root_id not in ("", scope.thread_root_message_id):
                raise ContextUnavailableError(
                    ContextUnavailableReason.ROOT_THREAD_BINDING
                )

    @staticmethod
    def _deduplicate(messages: list[ContextMessage]) -> list[ContextMessage]:
        output: list[ContextMessage] = []
        indexes: dict[str, int] = {}
        for message in messages:
            index = indexes.get(message.message_id)
            if index is None:
                indexes[message.message_id] = len(output)
                output.append(message)
                continue
            existing = output[index]
            existing_revision = MessageRevision.from_message(existing)
            incoming_revision = MessageRevision.from_message(message)
            if existing_revision.digest == incoming_revision.digest:
                continue
            if incoming_revision.update_time_ms > existing_revision.update_time_ms:
                output[index] = message
                continue
            if incoming_revision.update_time_ms < existing_revision.update_time_ms:
                continue
            raise ContextUnavailableError(ContextUnavailableReason.REVISION)
        return output

    @staticmethod
    def _revision_digest(messages: list[ContextMessage]) -> str:
        payload = [
            {
                "message_id": revision.message_id,
                "digest": revision.digest,
                "update_time_ms": revision.update_time_ms,
                "deleted": revision.deleted,
            }
            for revision in map(MessageRevision.from_message, messages)
        ]
        return _digest(payload)

    def _fetch_group_window(
        self,
        scope: EmployeeMessageScope,
        stable_thread: list[ContextMessage],
        deadline: float,
    ) -> list[ContextMessage]:
        current = stable_thread[-1]
        thread_ids = {message.message_id for message in stable_thread}
        eligible: list[ContextMessage] = []
        token = ""
        seen_tokens: set[str] = set()
        try:
            for page_number in range(1, self._config.max_pages + 1):
                page = self._checked_call(
                    lambda: self._source.list_chat_messages(
                        page_token=token,
                        page_size=self._config.group_page_size,
                    ),
                    deadline,
                )
                if not isinstance(page, MessagePage):
                    raise ContextUnavailableError(ContextUnavailableReason.SOURCE)
                for message in page.messages:
                    if message.chat_id != scope.chat_id:
                        raise ContextUnavailableError(
                            ContextUnavailableReason.SCOPE
                        )
                    if message.is_system or message.msg_type == "system":
                        continue
                    if self._not_after_current(message, current):
                        eligible.append(message)
                eligible = self._deduplicate(eligible)
                group_count = sum(
                    message.message_id not in thread_ids for message in eligible
                )
                if not page.has_more:
                    break
                cutoff = self._group_cutoff(eligible, thread_ids)
                page_times = [
                    message.create_time_ms for message in page.messages
                ]
                cohort_complete = (
                    group_count >= self._config.max_group_messages
                    and cutoff is not None
                    and page_times
                    and min(page_times) < cutoff
                )
                if cohort_complete:
                    break
                next_token = page.page_token
                if not next_token or next_token in seen_tokens:
                    raise ContextUnavailableError(
                        ContextUnavailableReason.PAGINATION
                    )
                seen_tokens.add(next_token)
                token = next_token
                if page_number == self._config.max_pages:
                    raise ContextUnavailableError(
                        ContextUnavailableReason.PAGINATION
                    )
            else:
                raise ContextUnavailableError(
                    ContextUnavailableReason.PAGINATION
                )
        finally:
            # Reset unconditionally: the deadline may expire after the source
            # accepted a page but before ``_checked_call`` returns it, so the
            # assembler cannot reliably infer continuation state here.
            self._source.reset_chat_traversal()

        thread_revisions = [
            message for message in eligible if message.message_id in thread_ids
        ]
        group_messages = [
            message for message in eligible if message.message_id not in thread_ids
        ]
        group_messages.sort(key=_group_order_key)
        return thread_revisions + group_messages[-self._config.max_group_messages :]

    def _group_cutoff(
        self,
        eligible: list[ContextMessage],
        thread_ids: set[str],
    ) -> int | None:
        group_messages = [
            message for message in eligible if message.message_id not in thread_ids
        ]
        if len(group_messages) < self._config.max_group_messages:
            return None
        group_messages.sort(key=_group_order_key, reverse=True)
        return group_messages[self._config.max_group_messages - 1].create_time_ms

    @staticmethod
    def _not_after_current(
        message: ContextMessage,
        current: ContextMessage,
    ) -> bool:
        if message.create_time_ms < current.create_time_ms:
            return True
        if message.create_time_ms > current.create_time_ms:
            return False
        if message.message_id == current.message_id:
            return True
        if (
            message.message_position is None
            or current.message_position is None
        ):
            return False
        if message.message_position == current.message_position:
            raise ContextUnavailableError(ContextUnavailableReason.ORDERING)
        return message.message_position < current.message_position

    def _reconcile_layers(
        self,
        scope: EmployeeMessageScope,
        stable_thread: list[ContextMessage],
        group: list[ContextMessage],
    ) -> tuple[list[ContextMessage], list[ContextMessage]]:
        thread = [
            message
            for message in stable_thread
            if not message.is_system and message.msg_type != "system"
        ]
        thread_indexes = {
            message.message_id: index for index, message in enumerate(thread)
        }
        retained_group: list[ContextMessage] = []
        for message in group:
            index = thread_indexes.get(message.message_id)
            if index is None:
                retained_group.append(replace(message, is_current=False))
                continue
            thread_message = thread[index]
            thread_revision = MessageRevision.from_message(thread_message)
            group_revision = MessageRevision.from_message(message)
            if thread_revision.digest == group_revision.digest:
                continue
            if group_revision.update_time_ms > thread_revision.update_time_ms:
                if not _same_message_identity(thread_message, message):
                    raise ContextUnavailableError(
                        ContextUnavailableReason.REVISION
                    )
                replacement = replace(
                    message,
                    is_current=message.message_id == scope.current_message_id,
                )
                if replacement.is_current and (
                    replacement.deleted
                    or replacement.is_system
                    or replacement.msg_type == "system"
                ):
                    raise ContextUnavailableError(
                        ContextUnavailableReason.CURRENT_MESSAGE
                    )
                thread[index] = replacement
                continue
            if group_revision.update_time_ms < thread_revision.update_time_ms:
                continue
            raise ContextUnavailableError(ContextUnavailableReason.REVISION)
        current_count = sum(message.is_current for message in thread)
        if current_count != 1:
            raise ContextUnavailableError(ContextUnavailableReason.CURRENT_MESSAGE)
        return thread, retained_group

    @staticmethod
    def _watermark(
        scope: EmployeeMessageScope,
        resolved: ResolvedThread,
        messages: list[ContextMessage],
    ) -> ThreadWatermark:
        last = messages[-1]
        return ThreadWatermark(
            thread_root_id=scope.thread_root_message_id,
            last_message_id=last.message_id,
            last_timestamp=last.timestamp,
            message_count=len(messages),
            revision=max(message.update_time_ms for message in messages),
            tenant_key=scope.tenant_key,
            chat_id=scope.chat_id,
            feishu_thread_id=resolved.feishu_thread_id,
            revision_digest=EmployeeThreadContext._revision_digest(messages),
        )

    def _apply_budget(
        self,
        *,
        scope: EmployeeMessageScope,
        watermark: ThreadWatermark,
        thread: list[ContextMessage],
        group: list[ContextMessage],
        l1_summary: str,
        l2_summary: str,
        reserve: int,
        constraints_digest: str,
    ) -> AssembledContext:
        source_thread = tuple(thread)
        source_group = tuple(group)
        source_l1 = l1_summary
        source_l2 = l2_summary
        budgeted = apply_context_budget(
            config=self._config,
            thread=thread,
            group=group,
            l1_summary=l1_summary,
            l2_summary=l2_summary,
            reserve=reserve,
        )
        l1_summary = budgeted.l1_summary
        l2_summary = budgeted.l2_summary
        total_chars = budgeted.total_chars
        total_tokens = budgeted.total_tokens
        trace = budgeted.trimming_trace
        layers_used = tuple(
            layer
            for layer, present in (
                (ContextLayer.THREAD_FULL, bool(thread)),
                (ContextLayer.GROUP_RECENT, bool(group)),
                (ContextLayer.L1_MEMORY, bool(l1_summary)),
                (ContextLayer.L2_GROUP, bool(l2_summary)),
            )
            if present
        )
        metrics = (
            _message_metrics(ContextLayer.THREAD_FULL, source_thread, thread),
            _message_metrics(ContextLayer.GROUP_RECENT, source_group, group),
            _memory_metrics(ContextLayer.L1_MEMORY, source_l1, l1_summary),
            _memory_metrics(ContextLayer.L2_GROUP, source_l2, l2_summary),
        )
        snapshot_hash = _digest(
            {
                "scope": {
                    "tenant_key": scope.tenant_key,
                    "agent_id": scope.agent_id,
                    "bot_principal_id": scope.bot_principal_id,
                    "app_id": scope.app_id,
                    "chat_id": scope.chat_id,
                    "root_id": scope.thread_root_message_id,
                    "current_id": scope.current_message_id,
                    "feishu_thread_id": watermark.feishu_thread_id,
                    "source_requester_principal_id": (
                        scope.source_requester_principal_id
                    ),
                    "authorization_scope": scope.authorization_scope.value,
                },
                "watermark": watermark.revision_digest,
                "thread": [
                    MessageRevision.from_message(message).digest for message in thread
                ],
                "group": [
                    MessageRevision.from_message(message).digest for message in group
                ],
                "l1_digest": hashlib.sha256(l1_summary.encode()).hexdigest(),
                "l2_digest": hashlib.sha256(l2_summary.encode()).hexdigest(),
                "reserve": reserve,
                "tokens_per_char": self._config.tokens_per_char,
                "constraints_digest": constraints_digest,
                "trace": [
                    (
                        record.layer.value,
                        record.removed_messages,
                        record.removed_chars,
                    )
                    for record in trace
                ],
            }
        )
        return AssembledContext(
            thread_messages=tuple(thread),
            group_messages=tuple(group),
            l1_summary=l1_summary,
            l2_summary=l2_summary,
            total_tokens_estimate=total_tokens,
            watermark=watermark,
            layers_used=layers_used,
            truncated=bool(trace),
            total_chars=total_chars,
            layer_metrics=metrics,
            trimming_trace=tuple(trace),
            snapshot_hash=snapshot_hash,
            system_prompt_tokens_reserved=reserve,
            constraints_digest=constraints_digest,
            tokens_per_char=self._config.tokens_per_char,
        )


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _group_order_key(message: ContextMessage) -> tuple[int, int, str]:
    """Order chat history without treating Thread-local position as global."""
    return (
        message.create_time_ms,
        message.message_position if message.message_position is not None else -1,
        message.message_id,
    )


def _same_message_identity(
    first: ContextMessage,
    second: ContextMessage,
) -> bool:
    """Check fields that a content revision is never allowed to rebind."""
    return all(
        getattr(first, field_name) == getattr(second, field_name)
        for field_name in (
            "message_id",
            "chat_id",
            "thread_id",
            "root_id",
            "parent_id",
            "sender_id",
            "sender_id_type",
            "sender_type",
            "sender_tenant_key",
            "msg_type",
            "create_time_ms",
            "message_position",
            "thread_message_position",
        )
    )


def _message_metrics(
    layer: ContextLayer,
    source: tuple[ContextMessage, ...],
    retained: list[ContextMessage],
) -> ContextLayerMetrics:
    return ContextLayerMetrics(
        layer=layer,
        source_messages=len(source),
        retained_messages=len(retained),
        source_chars=sum(len(message.text) for message in source),
        retained_chars=sum(len(message.text) for message in retained),
        omission_reason="budget" if len(retained) < len(source) else "",
    )


def _memory_metrics(
    layer: ContextLayer,
    source: str,
    retained: str,
) -> ContextLayerMetrics:
    return ContextLayerMetrics(
        layer=layer,
        source_messages=1 if source else 0,
        retained_messages=1 if retained else 0,
        source_chars=len(source),
        retained_chars=len(retained),
        omission_reason="budget" if source and not retained else "",
    )


__all__ = ["EmployeeThreadContext"]
