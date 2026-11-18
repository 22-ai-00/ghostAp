import os as _os

if _os.environ.get("GHOSTAP_DISABLE_LAZY_LARK", "").strip().lower() not in {"1", "true", "yes"}:
    try:
        if __package__ == "feishu":
            from utils.lark_lazy import (
                install_lazy_lark_channel as _install_lazy_lark_channel,
            )
            from utils.lark_lazy import (
                install_lazy_lark_oapi as _install_lazy_lark_oapi,
            )
        else:
            from ..utils.lark_lazy import (
                install_lazy_lark_channel as _install_lazy_lark_channel,
            )
            from ..utils.lark_lazy import (
                install_lazy_lark_oapi as _install_lazy_lark_oapi,
            )
        _install_lazy_lark_oapi()
        _install_lazy_lark_channel()
    except Exception:
        pass

from .emoji import EmojiType
from .image_handler import FeishuImageHandler
from .message_cache import MessageCache
from .message_formatter import FeishuMessageFormatter
from .ws_client import EmojiReaction, FeishuWSClient

__all__ = [
    "FeishuWSClient",
    "FeishuMessageFormatter",
    "FeishuImageHandler",
    "EmojiType",
    "EmojiReaction",
    "MessageCache",
]
