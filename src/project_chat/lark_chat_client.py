"""Feishu Chat API wrapper for project-chat binding."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .errors import CreateChatError

logger = logging.getLogger(__name__)


@dataclass
class CreateChatResult:
    """Result of creating a Feishu group chat."""
    chat_id: str
    name: str


class ManagedChatValidation(str, Enum):
    """Remote provenance result; UNKNOWN always fails closed."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class LarkChatClient:
    """Wraps Feishu IM v1 Chat API with retry and error handling.

    Follows the same retry/backoff pattern as FeishuIMClient._execute_with_retry.
    """

    def __init__(self, api_client_factory: Callable[[], Any], max_retries: int = 3):
        self._api_client_factory = api_client_factory
        self._max_retries = max_retries

    def create_chat(
        self,
        *,
        name: str,
        description: str,
        user_id_list: list[str],
    ) -> CreateChatResult:
        """Create a Feishu group chat. Bot is auto-added as creator.

        Raises CreateChatError on failure.
        """
        from lark_oapi.api.im.v1 import CreateChatRequest, CreateChatRequestBody

        client = self._api_client_factory()
        body = CreateChatRequestBody.builder() \
            .name(name) \
            .description(description) \
            .user_id_list(user_id_list) \
            .chat_mode("group") \
            .chat_type("private") \
            .build()
        request = CreateChatRequest.builder() \
            .user_id_type("open_id") \
            .request_body(body) \
            .build()

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = client.im.v1.chat.create(request)
                if response.success():
                    return CreateChatResult(
                        chat_id=response.data.chat_id,
                        name=name,
                    )
                last_error = f"[{response.code}] {response.msg}"
                if response.code in (230001, 230020, 99991672):
                    break
            except Exception as e:
                last_error = str(e)
            if attempt < self._max_retries - 1:
                time.sleep(0.3 * (2 ** attempt))

        raise CreateChatError(f"建群失败: {last_error}")

    def add_managers(self, chat_id: str, manager_ids: list[str]) -> None:
        """Add managers to a group chat (best-effort).

        Does NOT raise on failure — only logs warning.
        """
        from lark_oapi.api.im.v1 import (
            AddManagersChatManagersRequest,
            AddManagersChatManagersRequestBody,
        )

        client = self._api_client_factory()
        body = AddManagersChatManagersRequestBody.builder() \
            .manager_ids(manager_ids) \
            .build()
        request = AddManagersChatManagersRequest.builder() \
            .chat_id(chat_id) \
            .member_id_type("open_id") \
            .request_body(body) \
            .build()

        try:
            response = client.im.v1.chat_managers.add_managers(request)
            if not response.success():
                logger.warning(
                    "add_managers(%s) failed: [%s] %s",
                    chat_id[:12], response.code, response.msg,
                )
        except Exception as e:
            logger.warning("add_managers(%s) exception: %s", chat_id[:12], e)

    def validate_managed_chat(
        self,
        chat_id: str,
        owner_open_id: str,
    ) -> ManagedChatValidation:
        """Verify the receiving bot and configured Owner are both members."""

        from lark_oapi.api.im.v1 import (
            GetChatMembersRequest,
            IsInChatChatMembersRequest,
        )

        try:
            client = self._api_client_factory()
            bot_request = (
                IsInChatChatMembersRequest.builder().chat_id(chat_id).build()
            )
            bot_response = client.im.v1.chat_members.is_in_chat(bot_request)
            if not bot_response.success():
                return ManagedChatValidation.UNKNOWN
            if getattr(bot_response.data, "is_in_chat", None) is not True:
                return ManagedChatValidation.INVALID

            page_token = ""
            while True:
                builder = (
                    GetChatMembersRequest.builder()
                    .chat_id(chat_id)
                    .member_id_type("open_id")
                    .page_size(100)
                )
                if page_token:
                    builder = builder.page_token(page_token)
                response = client.im.v1.chat_members.get(builder.build())
                if not response.success():
                    return ManagedChatValidation.UNKNOWN
                data = response.data
                if getattr(data, "trigger_security_conf_limit", False):
                    return ManagedChatValidation.UNKNOWN
                if any(
                    getattr(member, "member_id", None) == owner_open_id
                    for member in (getattr(data, "items", None) or ())
                ):
                    return ManagedChatValidation.VALID
                if not getattr(data, "has_more", False):
                    return ManagedChatValidation.INVALID
                page_token = getattr(data, "page_token", "") or ""
                if not page_token:
                    return ManagedChatValidation.UNKNOWN
        except Exception:
            logger.exception("validate_managed_chat(%s) failed", chat_id[:12])
            return ManagedChatValidation.UNKNOWN

    def delete_chat(self, chat_id: str) -> bool | None:
        """Delete a Feishu group chat (best-effort, for rollback).

        Does not raise; returns whether Feishu confirmed deletion.
        """
        from lark_oapi.api.im.v1 import DeleteChatRequest

        try:
            client = self._api_client_factory()
            request = DeleteChatRequest.builder().chat_id(chat_id).build()
            response = client.im.v1.chat.delete(request)
            if not response.success():
                message = str(response.msg or "").strip().casefold()
                already_absent = any(
                    marker in message
                    for marker in (
                        "not found",
                        "not exist",
                        "does not exist",
                        "already deleted",
                        "不存在",
                        "已删除",
                    )
                )
                logger.warning(
                    "delete_chat(%s) failed: [%s] %s",
                    chat_id[:12], response.code, response.msg,
                )
                return True if already_absent else False
            return True
        except Exception as e:
            logger.warning("delete_chat(%s) exception: %s", chat_id[:12], e)
            return None

    def patch_description(self, chat_id: str, description: str) -> None:
        """Update group chat description (best-effort)."""
        from lark_oapi.api.im.v1 import UpdateChatRequest, UpdateChatRequestBody

        client = self._api_client_factory()
        body = UpdateChatRequestBody.builder().description(description).build()
        request = UpdateChatRequest.builder() \
            .chat_id(chat_id) \
            .request_body(body) \
            .build()

        try:
            response = client.im.v1.chat.update(request)
            if not response.success():
                logger.warning(
                    "patch_description(%s) failed: [%s] %s",
                    chat_id[:12], response.code, response.msg,
                )
        except Exception as e:
            logger.warning("patch_description(%s) exception: %s", chat_id[:12], e)
