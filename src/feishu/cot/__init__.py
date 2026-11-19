"""Task-scoped Feishu IM COT transport for programming process events.

Feishu's Python SDK does not expose typed ``message_cot`` resources in the
version used by GhostAP.  This package therefore uses the SDK's generic
``Client.request(BaseRequest)`` entry point while retaining the same strict
response, outbound-audit, and managed-scope fences as the existing message
and card transports.
"""

from __future__ import annotations

from .api_client import FeishuCOTAPIClient
from .session import FeishuCOTSession
from .stream import FeishuCOTStream
from .wire import COTBinding, COTTransportError

__all__ = [
    "COTBinding",
    "COTTransportError",
    "FeishuCOTAPIClient",
    "FeishuCOTSession",
    "FeishuCOTStream",
]
