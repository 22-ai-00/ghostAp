from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

import src.acp.session as acp_session_mod
import src.acp.telemetry as telemetry_mod
from src.acp.manager import ACPSessionManager
from src.acp.telemetry import (
    IdleHealthConfig,
    IdleHealthTelemetryContext,
)
from src.acp.telemetry import (
    _DefaultIdleHealthTelemetry as DefaultIdleHealthTelemetry,
)
from src.acp.telemetry import (
    _IdleHealthServiceProtocol as IdleHealthServiceProtocol,
)
from src.utils.time_ago import IdleHealth


def _make_mock_session(*, last_active: float = 0.0, server_running: bool = True) -> MagicMock:
    session = MagicMock()
    session.last_active = last_active
    session.is_server_running.return_value = server_running
    session.session_id = "mock-sid-001"
    session.message_count = 1
    session.to_snapshot.return_value = {"id": "mock-sid-001"}
    session.close.return_value = None
    return session


def test_ambiguous_resume_uses_remaining_deadline_and_is_closed_before_rethrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []
    resume_timeouts: list[float] = []
    close_deadlines: list[float] = []
    clock = {"now": 100.0}

    class UncertainResumeSession:
        def __init__(self) -> None:
            self.session_id = "new-session"
            self.message_count = 0
            self.last_active = time.time()
            self._force_dead = False

        def describe_agent(self) -> str:
            return "codex-acp"

        def load_session(self, _session_id: str, *, timeout: float) -> None:
            resume_timeouts.append(timeout)
            clock["now"] += 1.0
            raise ConnectionError("resume transport lost")

        def close(self) -> None:
            closed.append(self.session_id)

    created = UncertainResumeSession()
    def start_backend(**_kwargs):
        clock["now"] += 4.0
        return created, created.session_id, {}

    monkeypatch.setattr(
        "src.acp.manager.time.monotonic",
        lambda: clock["now"],
    )
    manager = ACPSessionManager(
        "codex",
        keepalive_interval=0,
        session_starter=start_backend,
    )
    original_close = manager._close_session_before

    def record_close(session, *, key: str, deadline: float) -> None:
        close_deadlines.append(deadline)
        session.close()

    manager._close_session_before = record_close  # type: ignore[method-assign]
    try:
        with pytest.raises(ConnectionError, match="resume transport lost"):
            manager.start_session(
                "chat-resume",
                cwd=".",
                session_id="target-session",
                startup_timeout=10,
            )
        assert manager.get_session("chat-resume") is None
        assert closed == ["new-session"]
        assert resume_timeouts == [pytest.approx(3.0)]
        assert close_deadlines == [pytest.approx(110.0)]
    finally:
        manager._close_session_before = original_close  # type: ignore[method-assign]
        manager.cleanup_all()


def test_resume_timeout_preserves_close_reserve_inside_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 200.0}
    resume_timeouts: list[float] = []
    close_remaining: list[float] = []
    closed: list[str] = []

    class TimedOutResumeSession:
        def __init__(self) -> None:
            self.session_id = "new-session"
            self.message_count = 0
            self.last_active = time.time()
            self._force_dead = False

        def describe_agent(self) -> str:
            return "codex-acp"

        def load_session(self, _session_id: str, *, timeout: float) -> None:
            resume_timeouts.append(timeout)
            clock["now"] += timeout
            raise TimeoutError("resume load deadline")

        def close(self) -> None:
            closed.append(self.session_id)

    created = TimedOutResumeSession()
    monkeypatch.setattr(
        "src.acp.manager.time.monotonic",
        lambda: clock["now"],
    )
    manager = ACPSessionManager(
        "codex",
        keepalive_interval=0,
        session_starter=lambda **_kwargs: (created, created.session_id, {}),
    )
    original_close = manager._close_session_before

    def record_close(session, *, key: str, deadline: float) -> None:
        del key
        close_remaining.append(deadline - clock["now"])
        if deadline <= clock["now"]:
            raise TimeoutError("close registration budget exhausted")
        session.close()

    manager._close_session_before = record_close  # type: ignore[method-assign]
    try:
        with pytest.raises(TimeoutError, match="resume load deadline"):
            manager.start_session(
                "chat-timeout-resume",
                cwd=".",
                session_id="target-session",
                startup_timeout=10,
            )
        assert resume_timeouts == [pytest.approx(5.0)]
        assert close_remaining == [pytest.approx(5.0)]
        assert closed == ["new-session"]
        assert manager.get_session("chat-timeout-resume") is None
    finally:
        manager._close_session_before = original_close  # type: ignore[method-assign]
        manager.cleanup_all()


def test_explicit_resume_rejection_is_the_only_new_session_fallback() -> None:
    rejection_type = getattr(acp_session_mod, "ACPResumeRejected", None)
    assert rejection_type is not None
    closed: list[str] = []

    class ExplicitlyRejectedSession:
        def __init__(self) -> None:
            self.session_id = "new-session"
            self.message_count = 0
            self.last_active = time.time()
            self._force_dead = False

        def describe_agent(self) -> str:
            return "codex-acp"

        def load_session(self, _session_id: str, *, timeout: float) -> None:
            assert timeout > 0
            raise rejection_type("resume explicitly rejected")

        def load_local_history(self, _session_id: str) -> list[dict]:
            return []

        def close(self) -> None:
            closed.append(self.session_id)

    created = ExplicitlyRejectedSession()
    manager = ACPSessionManager(
        "codex",
        keepalive_interval=0,
        session_starter=lambda **_kwargs: (created, created.session_id, {}),
    )
    try:
        session = manager.start_session(
            "chat-explicit-reject",
            cwd=".",
            session_id="missing-target",
        )
        assert session is created
        assert session.session_id == "new-session"
        assert session._force_dead is False
        assert closed == []
    finally:
        manager.cleanup_all()


class TestKeepaliveThreadLifecycle:
    def test_keepalive_thread_starts_when_interval_positive(self):
        mgr = ACPSessionManager("coco", keepalive_interval=1)
        try:
            assert mgr._keepalive_thread is not None
            assert mgr._keepalive_thread.is_alive()
            assert mgr._keepalive_thread.daemon is True
        finally:
            mgr.cleanup_all()
            assert mgr._keepalive_thread is None

    def test_keepalive_no_thread_when_interval_zero(self):
        mgr = ACPSessionManager("coco", keepalive_interval=0)
        try:
            assert mgr._keepalive_thread is None
        finally:
            mgr.cleanup_all()

    def test_cleanup_all_stops_keepalive_thread(self):
        mgr = ACPSessionManager("coco", keepalive_interval=1)
        t = mgr._keepalive_thread
        assert t is not None
        assert t.is_alive()
        mgr.cleanup_all()
        assert not t.is_alive()
        assert mgr._keepalive_thread is None


class TestKeepaliveSessionCleanup:
    def test_keepalive_cleans_dead_session(self):
        mgr = ACPSessionManager("coco", keepalive_interval=0.05, idle_healthcheck_s=0)
        try:
            session = _make_mock_session(last_active=time.time() - 300, server_running=False)
            key = "chat1:proj1"
            with mgr._lock:
                mgr._sessions[key] = session

            deadline = time.time() + 3
            while time.time() < deadline:
                with mgr._lock:
                    if key not in mgr._sessions:
                        break
                time.sleep(0.02)

            with mgr._lock:
                assert key not in mgr._sessions
            session.is_server_running.assert_called()
        finally:
            mgr.cleanup_all()

    def test_keepalive_keeps_active_session(self):
        mgr = ACPSessionManager("coco", keepalive_interval=0.05, idle_healthcheck_s=0)
        try:
            session = _make_mock_session(last_active=time.time() - 300, server_running=True)
            key = "chat2:proj2"
            with mgr._lock:
                mgr._sessions[key] = session

            time.sleep(0.3)

            with mgr._lock:
                assert key in mgr._sessions
            session.is_server_running.assert_called()
        finally:
            mgr.cleanup_all()


class TestSessionKeyEncodingDecoding:
    def test_session_key_roundtrip_default_project(self):
        mgr = ACPSessionManager("coco")
        try:
            key = mgr._session_key("chat-default")
            chat_id, project_id, thread_id = ACPSessionManager._parse_session_key(key)

            assert chat_id == "chat-default"
            # 默认项目应当被折叠为 None，而不是暴露占位符
            assert project_id is None
            assert thread_id is None
        finally:
            mgr.cleanup_all()

    def test_session_key_roundtrip_with_project_and_thread(self):
        mgr = ACPSessionManager("coco")
        try:
            key = mgr._session_key("chat-ctx", project_id="proj-ctx", thread_id="thread-ctx")

            chat_id, project_id, thread_id = ACPSessionManager._parse_session_key(key)

            assert chat_id == "chat-ctx"
            assert project_id == "proj-ctx"
            assert thread_id == "thread-ctx"
        finally:
            mgr.cleanup_all()

    def test_parse_session_key_handles_empty_and_non_string(self):
        # 空字符串
        chat_id, project_id, thread_id = ACPSessionManager._parse_session_key("")
        assert chat_id == ""
        assert project_id is None
        assert thread_id is None

        # 非字符串输入（例如整数）应被安全地转换为字符串后解析
        chat_id2, project_id2, thread_id2 = ACPSessionManager._parse_session_key(12345)  # type: ignore[arg-type]
        assert isinstance(chat_id2, str)
        assert project_id2 is None
        assert thread_id2 is None

    def test_parse_session_key_handles_minimal_legacy_key(self):
        # 只有 chat_id 一段的历史 key 应被视为「无 project/thread」
        chat_id, project_id, thread_id = ACPSessionManager._parse_session_key("legacy-chat-only")
        assert chat_id == "legacy-chat-only"
        assert project_id is None
        assert thread_id is None
class TestListActiveSessionsTimeAgo:
    def test_list_active_sessions_exposes_idle_bucket_only(self, monkeypatch):
        """list_active_sessions 应该返回 idle_seconds/idle_bucket 等结构化字段，由上层决定文案。"""

        # 构造一个假的会话对象，模拟 SyncSessionSnapshot 的必要属性
        class DummySession:
            def __init__(self, last_active: float, message_count: int = 3) -> None:
                self.session_id = "session-1"
                self.last_active = last_active
                self.message_count = message_count

            def to_snapshot(self) -> dict:
                return {"session_id": self.session_id}

            def close(self) -> None:
                return None

        # 使用默认 telemetry，验证正常路径下 idle_bucket/idle_health 暴露语义不变。
        mgr = ACPSessionManager(
            agent_type="coco",
            idle_health_config=IdleHealthConfig(
                idle_health_telemetry=DefaultIdleHealthTelemetry(),
            ),
        )

        now = time.time()
        dummy = DummySession(last_active=now - 300)  # 5 分钟前

        key = mgr._session_key("chat-id")
        with mgr._lock:
            mgr._sessions[key] = dummy  # type: ignore[assignment]

        try:
            results = mgr.list_active_sessions()
        finally:
            mgr.cleanup_all()

        assert len(results) == 1
        info = results[0]

        from src.utils.time_ago import compute_time_ago_bucket

        idle_seconds = info["idle_seconds"]
        assert idle_seconds >= 0

        # 验证 idle_bucket 语义与 idle_seconds 一致
        assert info["idle_bucket"] == compute_time_ago_bucket(idle_seconds)

        # 健康分类应返回 IdleHealth 枚举成员，并在 list_active_sessions 结果中以枚举形式暴露
        health = ACPSessionManager.classify_idle_health(info["idle_bucket"])
        assert isinstance(health, IdleHealth)
        assert health in {
            IdleHealth.HEALTHY,
            IdleHealth.IDLE,
            IdleHealth.STALE,
            IdleHealth.UNKNOWN,
        }
        assert "idle_health" in info
        assert isinstance(info["idle_health"], IdleHealth)
        assert info["idle_health"] == health


class DummyIdleHealthService(IdleHealthServiceProtocol):
    """用于验证 ACPSessionManager 通过协议依赖注入 IdleHealthService 的假实现。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def classify_session_idle_health(
        self,
        *,
        manager_agent_type: str,
        session_key: str,
        session_id: str,
        last_active: float,
        now: float | None = None,
        message_count: int | None = None,
    ):
        from src.utils.time_ago import IdleHealth, compute_time_ago_bucket

        eff_now = float(now or time.time())
        idle_seconds = max(0.0, eff_now - float(last_active or 0.0))
        bucket = compute_time_ago_bucket(idle_seconds)
        ctx: IdleHealthTelemetryContext = {
            "manager_agent_type": manager_agent_type,
            "session_key": session_key,
            "session_id": session_id,
            "idle_seconds": idle_seconds,
            "idle_bucket": bucket,
        }
        self.calls.append(ctx)
        # 固定返回 IDLE，便于断言 idle_health 字段来源于注入实现
        return IdleHealth.IDLE, bucket, idle_seconds, ctx


class TestIdleHealthServiceInjection:
    def test_list_active_sessions_uses_injected_idle_health_service(self):
        """当注入自定义 IdleHealthService 实现时，list_active_sessions 应使用该实现的结果。"""

        mgr = ACPSessionManager(
            agent_type="coco",
            idle_health_config=IdleHealthConfig(
                idle_health_service=DummyIdleHealthService(),
            ),
        )

        class DummySession:
            def __init__(self, last_active: float, message_count: int = 2) -> None:
                self.session_id = "sess-injected-1"
                self.last_active = last_active
                self.message_count = message_count

            def to_snapshot(self) -> dict:
                return {"session_id": self.session_id}

            def close(self) -> None:
                return None

        now = time.time()
        dummy = DummySession(last_active=now - 120)

        key = mgr._session_key("chat-injected", project_id="proj-injected", thread_id="thread-injected")
        with mgr._lock:
            mgr._sessions[key] = dummy  # type: ignore[assignment]

        try:
            results = mgr.list_active_sessions()
        finally:
            mgr.cleanup_all()

        assert len(results) == 1
        info = results[0]

        from src.utils.time_ago import IdleHealth

        # idle_health 字段应为注入实现固定返回的 IdleHealth.IDLE
        assert isinstance(info["idle_health"], IdleHealth)
        assert info["idle_health"] is IdleHealth.IDLE

    def test_list_active_sessions_passes_routing_context_to_idle_health(self, monkeypatch):
        """list_active_sessions 在 IdleHealth 回退路径中应传递包含路由信息的上下文。"""

        captured: list[tuple[dict, IdleHealthTelemetryContext | None]] = []

        class CapturingTelemetry(DefaultIdleHealthTelemetry):
            def log_idle_health_classification_fallback(self, *, bucket, error, context=None):  # type: ignore[override]
                captured.append((bucket, context or {}))


        class DummySession:
            def __init__(self, last_active: float, message_count: int = 3) -> None:
                self.session_id = "session-ctx-1"
                self.last_active = last_active
                self.message_count = message_count

            def to_snapshot(self) -> dict:
                return {"session_id": self.session_id}

            def close(self) -> None:
                return None

        # 通过让 SSOT helper 抛出可预期异常，触发 IdleHealth UNKNOWN 回退路径，
        # 从而验证 ACPSessionManager 在调用 telemetry 时传递了完整路由上下文。
        import src.utils.time_ago as time_ago_mod

        def _boom(_bucket):  # type: ignore[unused-argument]
            raise ValueError("bad bucket for keepalive test")

        monkeypatch.setattr(time_ago_mod, "classify_idle_health_from_bucket", _boom)

        mgr = ACPSessionManager(
            agent_type="coco",
            idle_health_config=IdleHealthConfig(
                idle_health_telemetry=CapturingTelemetry(),
            ),
        )

        now = time.time()
        dummy = DummySession(last_active=now - 60)

        key = mgr._session_key("chat-ctx", "proj-ctx", thread_id="thread-ctx")
        with mgr._lock:
            mgr._sessions[key] = dummy  # type: ignore[assignment]

        try:
            results = mgr.list_active_sessions()
        finally:
            mgr.cleanup_all()

        # classify_idle_health 应收到带完整路由信息的上下文
        assert len(captured) == 1
        bucket, ctx = captured[0]
        assert isinstance(ctx, dict)
        assert ctx.get("manager_agent_type") == "coco"
        assert ctx.get("session_key") == key
        assert ctx.get("session_id") == "session-ctx-1"
        assert ctx.get("chat_id") == "chat-ctx"
        # 显式 project_id 应被解析出来
        assert ctx.get("project_id") == "proj-ctx"
        # 线程维度同样应保留
        assert ctx.get("thread_id") == "thread-ctx"

        # list_active_sessions 返回结构中也应暴露解析后的字段
        assert len(results) == 1
        info = results[0]
        assert info["session_key"] == key
        assert info.get("session_id") == "session-ctx-1"


class TestIdleHealthTelemetryModule:
    def test_default_telemetry_delegates_to_module_hooks(self, monkeypatch):
        """DefaultIdleHealthTelemetry 应通过模块级 hook 实现真实逻辑，便于 monkeypatch。"""

        called: dict[str, object] = {}

        def fake_record_idle_health_fallback_metric(*, error_type: str) -> None:
            called["record"] = error_type

        def fake_log_idle_health_classification_fallback(*, bucket, error, context=None) -> None:  # type: ignore[no-untyped-def]
            called["log"] = (bucket, error, context)

        monkeypatch.setattr(telemetry_mod, "_record_idle_health_fallback_metric", fake_record_idle_health_fallback_metric)
        monkeypatch.setattr(telemetry_mod, "_log_idle_health_classification_fallback", fake_log_idle_health_classification_fallback)

        t = telemetry_mod._DefaultIdleHealthTelemetry()

        t.record_idle_health_fallback_metric(error_type="ValueError")
        t.log_idle_health_classification_fallback(bucket={"dummy": True}, error=RuntimeError("boom"), context={"foo": "bar"})

        assert called["record"] == "ValueError"
        bucket, err, ctx = called["log"]  # type: ignore[misc]
        assert isinstance(err, RuntimeError)
        assert isinstance(ctx, dict)
        assert ctx.get("foo") == "bar"

    def test_module_hooks_are_safe_and_emit_warning_log(self, caplog):
        """模块级 hook 应该永不抛异常，并在需要时输出 warning 日志。"""

        caplog.set_level("WARNING")

        telemetry_mod._record_idle_health_fallback_metric(error_type="TypeError")

        telemetry_mod._log_idle_health_classification_fallback(
            bucket="test-bucket",
            error=ValueError("bad bucket"),
            context={"manager_agent_type": "coco", "session_key": "chat:proj"},
        )

        # 不应抛异常，且至少有一条与 IdleHealth 回退相关的 warning 日志
        assert any("IdleHealth classification fallback" in r.getMessage() for r in caplog.records)

    def test_get_manager_compat_idle_health_telemetry_factory(self):
        """get_manager_compat_idle_health_telemetry 应返回满足协议的 Telemetry 实例。"""

        t = telemetry_mod._get_manager_compat_idle_health_telemetry()

        # 仅通过鸭子类型检查关键方法是否存在，避免对具体实现类型做过强绑定。
        assert hasattr(t, "record_idle_health_fallback_metric")
        assert hasattr(t, "log_idle_health_classification_fallback")

    def test_get_idle_health_telemetry_for_manager_prefers_explicit_instance(self):
        """get_idle_health_telemetry_for_manager 应优先返回显式传入的 telemetry 实例。"""

        class CustomTelemetry(DefaultIdleHealthTelemetry):
            pass

        custom = CustomTelemetry()

        result = telemetry_mod._get_idle_health_telemetry_for_manager(custom)

        assert result is custom

    def test_get_idle_health_telemetry_for_manager_falls_back_to_manager_factory(self, monkeypatch):
        """当未显式传入 telemetry 时，应回退到 manager 专用工厂。"""

        marker = object()

        def fake_get_manager_compat_idle_health_telemetry():  # type: ignore[no-untyped-def]
            return marker

        monkeypatch.setattr(
            telemetry_mod,
            "_get_manager_compat_idle_health_telemetry",
            fake_get_manager_compat_idle_health_telemetry,
        )

        result = telemetry_mod._get_idle_health_telemetry_for_manager(None)

        assert result is marker


class TestManagerIdleHealthDeprecation:
    def test_explicit_idle_health_args_emit_deprecation_log(self, caplog):
        """显式传入 idle_health_* 协作者参数时，应打印一次软 deprecate 日志。"""

        caplog.set_level("WARNING")

        # 仅通过 idle_health_telemetry 显式注入协作者，触发 __init__ 中的 deprecate 日志。
        _ = ACPSessionManager(agent_type="coco", idle_health_telemetry=DefaultIdleHealthTelemetry())

        messages = [r.getMessage() for r in caplog.records]
        assert any("ACPSessionManager.__init__ 的 idle_health_telemetry" in msg for msg in messages)

    def test_classify_idle_health_with_fallback_normal_path(self, monkeypatch):
        """_classify_idle_health_with_fallback 在正常路径下应直接返回 SSOT 结果。"""

        import src.utils.time_ago as time_ago_mod

        called: dict[str, object] = {}

        def fake_classify_from_bucket(bucket):  # type: ignore[no-untyped-def]
            called["bucket"] = bucket
            return IdleHealth.HEALTHY

        monkeypatch.setattr(time_ago_mod, "classify_idle_health_from_bucket", fake_classify_from_bucket)

        bucket = {"dummy": True}

        # 使用默认 telemetry，只验证正常路径不会触发 UNKNOWN 回退
        health = telemetry_mod._classify_idle_health_with_fallback(bucket, context=None)

        assert called["bucket"] is bucket
        assert health is IdleHealth.HEALTHY

    def test_classify_idle_health_for_manager_delegates_to_fallback_with_manager_telemetry(self, monkeypatch):
        """_classify_idle_health_for_manager 应统一通过 manager 专用 Telemetry 调用 fallback 入口。"""

        bucket = {"dummy": True}
        context = {"foo": "bar"}

        telemetry_instance = object()

        called: dict[str, object] = {}

        def fake_get_manager_compat_idle_health_telemetry():
            return telemetry_instance

        def fake_classify_idle_health_with_fallback(bucket, context=None, telemetry=None):  # type: ignore[no-untyped-def]
            called["args"] = (bucket, context, telemetry)
            return IdleHealth.IDLE

        monkeypatch.setattr(
            telemetry_mod,
            "_get_manager_compat_idle_health_telemetry",
            fake_get_manager_compat_idle_health_telemetry,
        )
        monkeypatch.setattr(telemetry_mod, "_classify_idle_health_with_fallback", fake_classify_idle_health_with_fallback)

        health = telemetry_mod._classify_idle_health_for_manager(bucket, context=context)

        assert health is IdleHealth.IDLE
        assert called["args"] == (bucket, context, telemetry_instance)


class TestIdleHealthService:
    def test_classify_session_idle_health_normal_path(self, monkeypatch) -> None:
        """IdleHealthService 应在正常路径下复用 SSOT helper 并返回预期枚举。"""

        import src.utils.time_ago as time_ago_mod

        captured: dict[str, object] = {}

        def fake_classify_from_bucket(bucket):  # type: ignore[no-untyped-def]
            captured["bucket"] = bucket
            return IdleHealth.HEALTHY

        monkeypatch.setattr(time_ago_mod, "classify_idle_health_from_bucket", fake_classify_from_bucket)

        svc = telemetry_mod._IdleHealthService()

        now = time.time()
        last_active = now - 300  # 5 分钟前

        health, bucket, idle_seconds, ctx = svc.classify_session_idle_health(
            manager_agent_type="coco",
            session_key="chat-1:proj-1:t:thread-1",
            session_id="sess-1",
            last_active=last_active,
            now=now,
            message_count=3,
        )

        assert health is IdleHealth.HEALTHY
        assert idle_seconds >= 0
        # bucket 应与 idle_seconds → TimeAgoBucket SSOT 结果保持一致
        from src.utils.time_ago import compute_time_ago_bucket

        assert captured["bucket"] == bucket
        assert bucket == compute_time_ago_bucket(idle_seconds)
        # Telemetry 上下文中至少包含基础字段
        assert ctx["manager_agent_type"] == "coco"
        assert ctx["session_key"] == "chat-1:proj-1:t:thread-1"
        assert ctx["session_id"] == "sess-1"

    def test_classify_session_idle_health_fallback_uses_telemetry_context(self, monkeypatch) -> None:
        """在 SSOT helper 抛出预期异常时，应通过注入 Telemetry 记录回退上下文。"""

        import src.utils.time_ago as time_ago_mod

        def _boom(_bucket):  # type: ignore[no-untyped-def]
            raise ValueError("bad bucket for IdleHealthService test")

        monkeypatch.setattr(time_ago_mod, "classify_idle_health_from_bucket", _boom)

        class FakeTelemetry:
            def __init__(self) -> None:
                self.logged: list[tuple[object, object, dict | None]] = []
                self.metrics: list[str] = []

            def log_idle_health_classification_fallback(self, *, bucket, error, context=None):  # type: ignore[no-untyped-def]
                self.logged.append((bucket, error, context))

            def record_idle_health_fallback_metric(self, *, error_type: str) -> None:
                self.metrics.append(error_type)

        telemetry = FakeTelemetry()
        svc = telemetry_mod._IdleHealthService(telemetry=telemetry)

        now = time.time()
        last_active = now - 60
        key = "chat-inst:proj-inst:t:thread-inst"

        health, bucket, idle_seconds, ctx = svc.classify_session_idle_health(
            manager_agent_type="coco",
            session_key=key,
            session_id="sess-instance-1",
            last_active=last_active,
            now=now,
            message_count=5,
        )

        from src.utils.time_ago import IdleHealth  # type: ignore[reimported]

        assert health is IdleHealth.UNKNOWN
        # Telemetry 应被调用一次，并包含路由上下文
        assert telemetry.metrics == ["ValueError"]
        assert len(telemetry.logged) == 1
        _bucket, err, ctx_logged = telemetry.logged[0]
        assert isinstance(err, ValueError)
        assert isinstance(ctx_logged, dict)
        assert ctx_logged.get("manager_agent_type") == "coco"
        assert ctx_logged.get("session_key") == key
        assert ctx_logged.get("session_id") == "sess-instance-1"
        assert ctx_logged.get("chat_id") == "chat-inst"
        assert ctx_logged.get("project_id") == "proj-inst"
        assert ctx_logged.get("thread_id") == "thread-inst"


class TestSessionTelemetryAdapterIntegration:
    def test_session_telemetry_start_and_end_called_with_injected_adapter(self) -> None:
        """ACPSessionManager 应在会话启动与结束时调用注入的 TelemetryAdapter。"""

        events: dict[str, list[dict]] = {"start": [], "end": [], "failed": []}

        class FakeAdapter:
            def on_session_start(self, *, manager_agent_type, session_key, session_id, backend_kind, model_name):  # type: ignore[no-untyped-def]
                events["start"].append(
                    {
                        "manager_agent_type": manager_agent_type,
                        "session_key": session_key,
                        "session_id": session_id,
                        "backend_kind": backend_kind,
                        "model_name": model_name,
                    }
                )

            def on_session_start_failed(self, *, manager_agent_type, session_key, backend_kind, error, diagnostics=None):  # type: ignore[no-untyped-def]
                events["failed"].append({"manager_agent_type": manager_agent_type, "session_key": session_key, "backend_kind": backend_kind, "error": error, "diagnostics": diagnostics})

            def on_session_end(self, *, manager_agent_type, session_key, session_id, message_count, reason=None, extra=None):  # type: ignore[no-untyped-def]
                events["end"].append(
                    {
                        "manager_agent_type": manager_agent_type,
                        "session_key": session_key,
                        "session_id": session_id,
                        "message_count": message_count,
                        "reason": reason,
                        "extra": extra,
                    }
                )

        adapter = FakeAdapter()

        def starter(**_kwargs):  # type: ignore[no-untyped-def]
            class DummySession:
                def __init__(self) -> None:
                    self.session_id = "sess-start-1"
                    self.message_count = 0
                    self.last_active = time.time()

                def describe_agent(self) -> str:
                    return "dummy-agent"

                def load_session(self, _sid: str, timeout: float = 60) -> None:
                    del timeout
                    return None

                def load_local_history(self, _sid: str) -> None:
                    return None

                def to_snapshot(self) -> dict:
                    return {"session_id": self.session_id, "message_count": self.message_count}

                def close(self) -> None:
                    return None

            s = DummySession()
            return s, s.session_id, {}

        mgr = ACPSessionManager(
            "coco",
            session_starter=starter,
            idle_health_config=IdleHealthConfig(
                session_telemetry=adapter,
            ),
        )

        try:
            session = mgr.start_session("chat-telemetry", cwd=".")
            assert session.session_id == "sess-start-1"

            # 主动结束会话以触发 on_session_end
            mgr.end_session("chat-telemetry")
        finally:
            mgr.cleanup_all()

        assert len(events["start"]) == 1
        start_event = events["start"][0]
        assert start_event["manager_agent_type"] == "coco"
        assert start_event["session_id"] == "sess-start-1"
        assert start_event["backend_kind"] == "acp"

        assert len(events["end"]) == 1
        end_event = events["end"][0]
        assert end_event["manager_agent_type"] == "coco"
        assert end_event["session_id"] == "sess-start-1"
        assert isinstance(end_event["message_count"], int)

        # 成功路径下不应触发 start_failed 事件
        assert events["failed"] == []

    def test_session_telemetry_start_failed_called_on_handshake_failure(self, monkeypatch) -> None:
        """当内部启动逻辑在握手阶段失败时，应调用 on_session_start_failed。"""

        events: dict[str, list[dict]] = {"failed": []}

        class FakeAdapter:
            def on_session_start(self, **_kwargs):  # type: ignore[no-untyped-def]
                return None

            def on_session_start_failed(self, *, manager_agent_type, session_key, backend_kind, error, diagnostics=None):  # type: ignore[no-untyped-def]
                events["failed"].append(
                    {
                        "manager_agent_type": manager_agent_type,
                        "session_key": session_key,
                        "backend_kind": backend_kind,
                        "error": error,
                        "diagnostics": diagnostics,
                    }
                )

            def on_session_end(self, **_kwargs):  # type: ignore[no-untyped-def]
                return None

        adapter = FakeAdapter()

        import src.acp.manager as manager_mod
        from src.acp.startup_utils import StartupOperationalError

        class FailingSession:
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                self.session_id = ""

            def describe_agent(self) -> str:
                return "failing-agent"

            def start(self, startup_timeout: float) -> str:  # type: ignore[no-untyped-def]
                raise StartupOperationalError("handshake failed for telemetry test")

        # 拦截 SyncACPSession，避免真实进程启动
        monkeypatch.setattr(manager_mod, "SyncACPSession", FailingSession)

        mgr = ACPSessionManager(
            "coco",
            idle_health_config=IdleHealthConfig(
                session_telemetry=adapter,
            ),
        )

        try:
            with pytest.raises(RuntimeError):
                mgr.start_session("chat-fail", cwd=".")
        finally:
            mgr.cleanup_all()

        # 至少应记录一次 start_failed 事件
        assert len(events["failed"]) >= 1
        failed = events["failed"][0]
        assert failed["manager_agent_type"] == "coco"
        assert failed["backend_kind"] == "acp"
        assert "chat-fail" in failed["session_key"]
        assert isinstance(failed["error"], Exception)


class TestManagerInitIdleHealthConfigPriority:
    def test_manager_init_uses_config_when_no_explicit_overrides(self):
        """当未传显式参数时，应优先使用 IdleHealthConfig 中的协作者，再回退到默认工厂。"""

        from src.acp.telemetry import DefaultSessionTelemetryAdapter

        class CustomTelemetry(DefaultIdleHealthTelemetry):
            pass

        class CustomService(IdleHealthServiceProtocol):  # type: ignore[misc]
            def classify_session_idle_health(
                self,
                *,
                manager_agent_type: str,
                session_key: str,
                session_id: str,
                last_active: float,
                now: float | None = None,
                message_count: int | None = None,
            ):
                from src.utils.time_ago import IdleHealth, compute_time_ago_bucket

                bucket = compute_time_ago_bucket(0.0)
                return IdleHealth.HEALTHY, bucket, 0.0, {
                    "manager_agent_type": manager_agent_type,
                    "session_key": session_key,
                    "session_id": session_id,
                    "idle_seconds": 0.0,
                    "idle_bucket": bucket,
                }

        class CustomSessionTelemetry(DefaultSessionTelemetryAdapter):
            pass

        cfg = IdleHealthConfig(
            idle_health_telemetry=CustomTelemetry(),
            idle_health_service=CustomService(),
            session_telemetry=CustomSessionTelemetry(),
        )

        mgr = ACPSessionManager("coco", idle_health_config=cfg)
        try:
            # 私有字段仅用于测试解析结果，不作为公共 API 暴露
            assert isinstance(mgr._idle_health_telemetry, CustomTelemetry)
            assert isinstance(mgr._idle_health_service, CustomService)
            assert isinstance(mgr._session_telemetry, CustomSessionTelemetry)
        finally:
            mgr.cleanup_all()

    def test_explicit_args_override_idle_health_config(self):
        """显式传入的 Telemetry/Service/Adapter 应覆盖 IdleHealthConfig 中的配置。"""

        from src.acp.telemetry import DefaultSessionTelemetryAdapter

        class TelemetryA(DefaultIdleHealthTelemetry):
            pass

        class TelemetryB(DefaultIdleHealthTelemetry):
            pass

        class ServiceA(IdleHealthServiceProtocol):  # type: ignore[misc]
            def classify_session_idle_health(
                self,
                *,
                manager_agent_type: str,
                session_key: str,
                session_id: str,
                last_active: float,
                now: float | None = None,
                message_count: int | None = None,
            ):
                from src.utils.time_ago import IdleHealth, compute_time_ago_bucket

                bucket = compute_time_ago_bucket(0.0)
                return IdleHealth.IDLE, bucket, 0.0, {
                    "manager_agent_type": manager_agent_type,
                    "session_key": session_key,
                    "session_id": session_id,
                    "idle_seconds": 0.0,
                    "idle_bucket": bucket,
                }

        class ServiceB(ServiceA):
            pass

        class SessionTelemetryA(DefaultSessionTelemetryAdapter):
            pass

        class SessionTelemetryB(DefaultSessionTelemetryAdapter):
            pass

        cfg = IdleHealthConfig(
            idle_health_telemetry=TelemetryA(),
            idle_health_service=ServiceA(),
            session_telemetry=SessionTelemetryA(),
        )

        mgr = ACPSessionManager(
            "coco",
            idle_health_config=cfg,
            idle_health_telemetry=TelemetryB(),
            idle_health_service=ServiceB(),
            session_telemetry=SessionTelemetryB(),
        )

        try:
            assert isinstance(mgr._idle_health_telemetry, TelemetryB)
            assert isinstance(mgr._idle_health_service, ServiceB)
            assert isinstance(mgr._session_telemetry, SessionTelemetryB)
        finally:
            mgr.cleanup_all()
