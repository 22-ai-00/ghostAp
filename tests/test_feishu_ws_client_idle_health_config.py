from __future__ import annotations

from src.acp.telemetry import DefaultSessionTelemetryAdapter
from src.feishu.ws_client import FeishuWSClient


def _dummy_callback(message_id: str, chat_id: str, text: str, parent_id: str | None) -> None:
    # 测试用占位回调，不做任何实际发送
    return None


def test_feishu_ws_client_uses_default_acp_session_managers():
    """构造 FeishuWSClient 时应仍然能够无感初始化各 ACPSessionManager。"""

    client = FeishuWSClient(message_callback=_dummy_callback)

    # 仅做存在性与类型层面的 smoke 检查，避免真正启动 ACP/CLI 会话
    assert client._coco_manager is not None
    assert client._ttadk_manager is not None


def test_feishu_ws_client_injects_default_session_telemetry_adapter():
    """WS 客户端应通过 IdleHealthConfig 为各 ACPSessionManager 注入默认 Session Telemetry 适配器。"""

    client = FeishuWSClient(message_callback=_dummy_callback)

    # 各 manager 的会话生命周期 Telemetry 都应默认使用 DefaultSessionTelemetryAdapter
    assert isinstance(client._coco_manager._session_telemetry, DefaultSessionTelemetryAdapter)
    assert isinstance(client._claude_manager._session_telemetry, DefaultSessionTelemetryAdapter)
    assert isinstance(client._aiden_manager._session_telemetry, DefaultSessionTelemetryAdapter)
    assert isinstance(client._codex_manager._session_telemetry, DefaultSessionTelemetryAdapter)
    assert isinstance(client._gemini_manager._session_telemetry, DefaultSessionTelemetryAdapter)
    assert isinstance(client._ttadk_manager._session_telemetry, DefaultSessionTelemetryAdapter)
