import io
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from lark_oapi.api.im.v1 import (
    CreateFileRequest,
    CreateFileRequestBody,
    CreateImageRequest,
    CreateImageRequestBody,
    CreateMessageReactionRequest,
    CreateMessageReactionRequestBody,
    CreateMessageRequest,
    CreateMessageRequestBody,
    Emoji,
    GetMessageResourceRequest,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from ..utils.errors import LARK_CODE_MESSAGE_NOT_FOUND, LARK_CODE_MESSAGE_RECALLED, get_error_detail
from .emoji import EmojiReaction

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


def _sanitize_content(content: str) -> str:
    """Remove surrogate code points that cannot be encoded in UTF-8.

    lark_oapi internally calls JSON.marshal(body).encode('utf-8') which
    raises UnicodeEncodeError if the string contains unpaired surrogates
    (U+D800-U+DFFF). This helper replaces them with the Unicode
    replacement character U+FFFD so the message can still be delivered.
    """
    try:
        # Fast path: if encoding succeeds, no surrogates present
        content.encode("utf-8")
        return content
    except UnicodeEncodeError:
        # Encode allowing surrogates, then decode replacing them
        return content.encode("utf-8", errors="surrogatepass").decode(
            "utf-8", errors="replace"
        )


class FeishuIMClient:
    """Client for Feishu IM API interactions with retry logic."""

    def __init__(
        self,
        api_client_factory: Callable[[], Any],
        settings: "Settings",
        *,
        outbound_audit: Callable[[str, str, str], None] | None = None,
        outbound_audit_failure: Callable[[Exception], None] | None = None,
        tenant_key_resolver: Callable[[], str] | None = None,
    ) -> None:
        self.api_client_factory = api_client_factory
        self.settings = settings
        self._outbound_audit = outbound_audit
        self._outbound_audit_failure = outbound_audit_failure
        self._tenant_key_resolver = tenant_key_resolver

    def _audit_outbound(
        self,
        operation: str,
        target: str,
        *,
        tenant_key: str | None = None,
    ) -> None:
        audit = self._outbound_audit
        if audit is None:
            return
        resolved_tenant_key = tenant_key
        if resolved_tenant_key is None and self._tenant_key_resolver is not None:
            try:
                resolved = self._tenant_key_resolver()
                resolved_tenant_key = resolved if isinstance(resolved, str) else ""
            except Exception:
                resolved_tenant_key = ""
        if resolved_tenant_key is None:
            resolved_tenant_key = ""
        try:
            audit(resolved_tenant_key, operation, target)
        except Exception as exc:
            logger.error("main Bot outbound audit failed closed: %s", type(exc).__name__)
            if self._outbound_audit_failure is not None:
                try:
                    self._outbound_audit_failure(exc)
                except Exception:
                    logger.error("main Bot audit failure callback failed", exc_info=True)
            raise

    def _execute_with_retry(
        self, func: Callable[[], Any], action_name: str, max_retries: Optional[int] = None
    ) -> Optional[Any]:
        """Execute an API call with retry logic and error noise reduction."""
        if max_retries is None:
            max_retries = self.settings.im_api_max_retries

        for attempt in range(max_retries):
            try:
                response = func()
                # Check for success (lark_oapi response object has success() method)
                if hasattr(response, "success") and response.success():
                    return response

                # Handling for API failures
                code = getattr(response, "code", None)
                msg = getattr(response, "msg", "Unknown error")

                # Noise reduction for expected business errors
                # 230001: Message does not exist
                # 230020: The message has been recalled
                if code in (LARK_CODE_MESSAGE_NOT_FOUND, LARK_CODE_MESSAGE_RECALLED):
                    logger.info("%s中止(消息不存在或已撤回): %s - %s", action_name, code, msg)
                    # Return the failed response so caller knows it failed but due to expected reasons
                    return response

                logger.warning("%s失败(尝试%d/%d): %s - %s", action_name, attempt + 1, max_retries, code, msg)
                if code == 230099 and "ErrCode: 200621" in str(msg):
                    logger.warning("[METRIC] card_render_failed err_code=200621 action=%s", action_name)
            except Exception as e:
                logger.warning("%s异常(尝试%d/%d): %s", action_name, attempt + 1, max_retries, get_error_detail(e), exc_info=True)

            if attempt < max_retries - 1:
                time.sleep(0.3 * (2**attempt))

        return None

    def send_message(
        self,
        receive_id_type: str,
        receive_id: str,
        content: str,
        msg_type: str = "text",
        max_retries: Optional[int] = None,
    ) -> Optional[Any]:
        """Send a message."""
        content = _sanitize_content(content)
        self._audit_outbound("create", receive_id)
        client = self.api_client_factory()
        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder().receive_id(receive_id).content(content).msg_type(msg_type).build()
            )
            .build()
        )

        return self._execute_with_retry(lambda: client.im.v1.message.create(request), "发送消息", max_retries)

    def reply_message(
        self,
        message_id: str,
        content: str,
        msg_type: str = "text",
        reply_in_thread: bool = False,
        max_retries: Optional[int] = None,
        idempotency_key: str | None = None,
        audit_aliases: tuple[str, ...] | None = None,
        audit_tenant_key: str | None = None,
    ) -> Optional[Any]:
        """Reply to a message."""
        content = _sanitize_content(content)
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 50
        ):
            raise ValueError("invalid Feishu message UUID")
        if self._outbound_audit is not None:
            if (
                not isinstance(audit_aliases, tuple)
                or not audit_aliases
                or any(
                    not isinstance(alias, str) or not alias
                    for alias in audit_aliases
                )
            ):
                raise RuntimeError("main Bot reply recipient scope is unavailable")
            for target in dict.fromkeys((message_id, *audit_aliases)):
                self._audit_outbound(
                    "reply",
                    target,
                    tenant_key=audit_tenant_key,
                )
        client = self.api_client_factory()
        body = (
            ReplyMessageRequestBody.builder()
            .content(content)
            .msg_type(msg_type)
            .reply_in_thread(reply_in_thread)
        )
        if idempotency_key:
            body = body.uuid(idempotency_key)
        request = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(body.build())
            .build()
        )

        return self._execute_with_retry(lambda: client.im.v1.message.reply(request), "回复消息", max_retries)

    def upload_file(
        self,
        file_path: str,
        file_type: str = "stream",
        file_name: str | None = None,
        duration: int | None = None,
        max_retries: Optional[int] = None,
    ) -> Optional[str]:
        """Upload a local file to Feishu IM and return its file_key."""
        client = self.api_client_factory()
        resolved_name = file_name or os.path.basename(file_path)
        with open(file_path, "rb") as file_obj:
            body_builder = (
                CreateFileRequestBody.builder()
                .file_type(file_type)
                .file_name(resolved_name)
                .file(file_obj)
            )
            if duration is not None:
                body_builder.duration(duration)
            request = CreateFileRequest.builder().request_body(body_builder.build()).build()
            response = self._execute_with_retry(lambda: client.im.v1.file.create(request), "上传文件", max_retries)

        if response is None:
            return None
        if hasattr(response, "success") and not response.success():
            return None
        data = getattr(response, "data", None)
        file_key = getattr(data, "file_key", None)
        return str(file_key) if file_key else None

    def upload_image_bytes(
        self,
        image_bytes: bytes,
        *,
        image_type: str = "message",
        max_retries: Optional[int] = None,
    ) -> Optional[str]:
        """Upload in-memory image bytes to Feishu IM and return image_key."""
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("image_bytes must be non-empty bytes")
        client = self.api_client_factory()

        def _upload_once():
            with io.BytesIO(image_bytes) as image_file:
                request = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type(image_type)
                        .image(image_file)
                        .build()
                    )
                    .build()
                )
                return client.im.v1.image.create(request)

        response = self._execute_with_retry(
            _upload_once,
            "上传图片",
            max_retries,
        )

        if response is None:
            return None
        if hasattr(response, "success") and not response.success():
            return None
        data = getattr(response, "data", None)
        image_key = getattr(data, "image_key", None)
        return str(image_key) if image_key else None

    def reply_file(
        self,
        message_id: str,
        file_key: str,
        reply_in_thread: bool = False,
        max_retries: Optional[int] = None,
        audit_aliases: tuple[str, ...] | None = None,
    ) -> Optional[Any]:
        """Reply to a message with a Feishu file attachment."""
        content = _sanitize_content(json.dumps({"file_key": file_key}, ensure_ascii=False))
        if self._outbound_audit is not None:
            if (
                not isinstance(audit_aliases, tuple)
                or not audit_aliases
                or any(
                    not isinstance(alias, str) or not alias
                    for alias in audit_aliases
                )
            ):
                raise RuntimeError("main Bot reply recipient scope is unavailable")
            for target in dict.fromkeys((message_id, *audit_aliases)):
                self._audit_outbound("reply", target)
        client = self.api_client_factory()
        request = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .content(content)
                .msg_type("file")
                .reply_in_thread(reply_in_thread)
                .build()
            )
            .build()
        )

        return self._execute_with_retry(lambda: client.im.v1.message.reply(request), "回复文件", max_retries)

    def patch_message(
        self,
        message_id: str,
        content: str,
        max_retries: Optional[int] = None,
        *,
        audit_aliases: tuple[str, ...] | None = None,
    ) -> Optional[Any]:
        """Patch a message."""
        content = _sanitize_content(content)
        if self._outbound_audit is not None:
            if (
                not isinstance(audit_aliases, tuple)
                or not audit_aliases
                or any(
                    not isinstance(alias, str) or not alias
                    for alias in audit_aliases
                )
            ):
                raise RuntimeError("main Bot patch recipient scope is unavailable")
            for target in dict.fromkeys((message_id, *audit_aliases)):
                self._audit_outbound("patch", target)
        client = self.api_client_factory()
        request = (
            PatchMessageRequest.builder()
            .message_id(message_id)
            .request_body(PatchMessageRequestBody.builder().content(content).build())
            .build()
        )

        return self._execute_with_retry(lambda: client.im.v1.message.patch(request), "更新消息", max_retries)

    def add_reaction(self, message_id: str, emoji_type: str) -> None:
        """Add a reaction to a message."""
        if not EmojiReaction.should_send(emoji_type):
            logger.debug("跳过非保留自动表情: %s", emoji_type)
            return

        client = self.api_client_factory()
        request = (
            CreateMessageReactionRequest.builder()
            .message_id(message_id)
            .request_body(
                CreateMessageReactionRequestBody.builder()
                .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                .build()
            )
            .build()
        )

        self._execute_with_retry(lambda: client.im.v1.message_reaction.create(request), "添加表情")

    def get_resource(
        self, message_id: str, file_key: str, resource_type: str, max_retries: Optional[int] = None
    ) -> Optional[Any]:
        """Download a resource (image, file, etc.)."""
        client = self.api_client_factory()
        request = (
            GetMessageResourceRequest.builder().message_id(message_id).file_key(file_key).type(resource_type).build()
        )

        return self._execute_with_retry(
            lambda: client.im.v1.message_resource.get(request), f"下载资源({resource_type})", max_retries
        )
