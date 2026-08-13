import asyncio
import logging
import threading
import time
from typing import Optional

from lark_channel.ws import client as lark_ws_client_impl

from ..utils.errors import get_error_detail

logger = logging.getLogger(__name__)

class WSHealthMonitor:
    """Watchdog for Feishu WebSocket connection health."""

    def __init__(self, client_instance, settings):
        self._client_instance = client_instance
        self.settings = settings
        self._health_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._last_connect_at = 0.0
        self._last_frame_at = 0.0
        self._last_pong_at = 0.0
        self._reconnect_requested_at = 0.0
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

    def record_activity(self, kind: str) -> None:
        now = time.time()
        reconnect_outage: float | None = None
        with self._health_lock:
            if kind == "connected":
                if self._reconnect_requested_at > 0.0:
                    reconnect_outage = max(0.0, now - self._reconnect_requested_at)
                self._last_connect_at = now
                self._last_frame_at = now
                self._last_pong_at = now
                self._reconnect_requested_at = 0.0
            elif kind in {"pong", "ping", "control", "data"}:
                self._last_frame_at = now
                if kind == "pong":
                    self._last_pong_at = now
            elif kind == "disconnected" and self._reconnect_requested_at <= 0.0:
                self._reconnect_requested_at = now
                logger.warning("WS断连，已触发重连请求: ts=%.3f", now)
                logger.warning("[METRIC] ws_disconnect")
        if reconnect_outage is not None:
            logger.info("WS重连成功: outage_seconds=%.3f", reconnect_outage)
            logger.info(
                "[METRIC] ws_reconnected outage_seconds=%.3f",
                reconnect_outage,
            )

    def _get_watchdog_interval(self) -> float:
        value = getattr(self.settings, "feishu_ws_watchdog_interval", 15.0)
        try:
            return max(1.0, float(value))
        except Exception:
            return 15.0

    def _get_stale_timeout(self) -> float:
        configured = getattr(self.settings, "feishu_ws_stale_timeout", 300.0)
        try:
            configured_timeout = max(60.0, float(configured))
        except Exception:
            configured_timeout = 300.0

        ping_interval = 120.0
        client = getattr(self._client_instance, "_client", None)
        if client is not None:
            try:
                ping_interval = max(1.0, float(getattr(client, "_ping_interval", 120.0) or 120.0))
            except Exception:
                ping_interval = 120.0

        grace = getattr(self.settings, "feishu_ws_stale_grace_seconds", 30.0)
        try:
            grace_seconds = max(5.0, float(grace))
        except Exception:
            grace_seconds = 30.0

        return max(configured_timeout, ping_interval * 2 + grace_seconds)

    def _trigger_disconnect(self, *, reason: str) -> bool:
        client = getattr(self._client_instance, "_client", None)
        if client is None or getattr(client, "_conn", None) is None:
            return False

        try:
            fut = asyncio.run_coroutine_threadsafe(client._disconnect(), lark_ws_client_impl.loop)
            fut.result(timeout=5)
            logger.warning("飞书长连接 watchdog 已触发重连: %s", reason)
            return True
        except Exception as e:
            logger.warning("飞书长连接 watchdog 触发重连失败: reason=%s err=%s", reason, get_error_detail(e))
            return False

    def check_health_once(self, now: Optional[float] = None) -> bool:
        client = getattr(self._client_instance, "_client", None)
        if client is None or getattr(client, "_conn", None) is None:
            return False

        current_time = now if now is not None else time.time()
        stale_timeout = self._get_stale_timeout()

        with self._health_lock:
            last_seen = max(self._last_pong_at, self._last_frame_at, self._last_connect_at)
            if last_seen <= 0.0:
                return False
            idle_for = current_time - last_seen
            if idle_for <= stale_timeout:
                return False

            requested_at = self._reconnect_requested_at
            if requested_at > 0.0 and (current_time - requested_at) < 30.0:
                return False

            self._reconnect_requested_at = current_time

        return self._trigger_disconnect(reason=f"idle_for={idle_for:.1f}s > timeout={stale_timeout:.1f}s")

    def _watchdog_loop(self) -> None:
        interval = self._get_watchdog_interval()
        while not self._watchdog_stop.wait(interval):
            try:
                self.check_health_once()
            except Exception as e:
                logger.debug("飞书长连接 watchdog 检查失败: %s", get_error_detail(e))

    def start_watchdog(self) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="feishu_ws_watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2)
        self._watchdog_thread = None

    def disconnect(self) -> bool:
        """Stop primary WebSocket intake before application task draining."""

        return self._trigger_disconnect(reason="application shutdown")
