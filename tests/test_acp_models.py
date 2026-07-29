"""Tests for acp.models — data models and enums."""

import time

import src.acp.telemetry as telemetry_mod
from src.acp.models import ACPEvent, ACPEventType, ACPSessionState, PlanEntryInfo, PlanInfo, PromptResult, ToolCallInfo
from src.acp.renderer import render_prompt_result_markdown


class TestACPEventType:
    def test_values(self):
        assert ACPEventType.TEXT_CHUNK.value == "text_chunk"
        assert ACPEventType.THOUGHT_CHUNK.value == "thought_chunk"
        assert ACPEventType.TOOL_CALL_START.value == "tool_call_start"
        assert ACPEventType.TOOL_CALL_UPDATE.value == "tool_call_update"
        assert ACPEventType.TOOL_CALL_DONE.value == "tool_call_done"
        assert ACPEventType.PLAN_UPDATE.value == "plan_update"

    def test_all_values_unique(self):
        values = [e.value for e in ACPEventType]
        assert len(values) == len(set(values))


class TestToolCallInfo:
    def test_create(self):
        tc = ToolCallInfo(id="tc1", title="Read file", kind="read", status="completed")
        assert tc.id == "tc1"
        assert tc.title == "Read file"
        assert tc.kind == "read"
        assert tc.status == "completed"
        assert tc.content == ""
        assert tc.locations == []

    def test_with_locations(self):
        tc = ToolCallInfo(
            id="tc2", title="Edit", kind="edit", status="in_progress", locations=["/tmp/a.py", "/tmp/b.py"]
        )
        assert len(tc.locations) == 2


class TestPlanInfo:
    def test_empty_plan(self):
        plan = PlanInfo()
        assert plan.entries == []

    def test_plan_with_entries(self):
        plan = PlanInfo(
            entries=[
                PlanEntryInfo(content="step 1", status="completed"),
                PlanEntryInfo(content="step 2", status="in_progress"),
                PlanEntryInfo(content="step 3"),
            ]
        )
        assert len(plan.entries) == 3
        assert plan.entries[0].status == "completed"
        assert plan.entries[2].status == "pending"


class TestACPEvent:
    def test_text_event(self):
        event = ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="hello")
        assert event.event_type == ACPEventType.TEXT_CHUNK
        assert event.text == "hello"
        assert event.tool_call is None
        assert event.plan is None
        assert event.timestamp > 0

    def test_tool_call_event(self):
        tc = ToolCallInfo(id="tc1", title="Read", kind="read", status="completed")
        event = ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=tc)
        assert event.tool_call.id == "tc1"

    def test_plan_event(self):
        plan = PlanInfo(entries=[PlanEntryInfo(content="step 1")])
        event = ACPEvent(event_type=ACPEventType.PLAN_UPDATE, plan=plan)
        assert len(event.plan.entries) == 1


class TestACPSessionState:
    def test_create(self):
        state = ACPSessionState(session_id="s1", agent_type="coco", cwd="/tmp")
        assert state.session_id == "s1"
        assert state.agent_type == "coco"
        assert state.message_count == 0
        assert state.is_active

    def test_to_dict(self):
        state = ACPSessionState(session_id="s1", agent_type="claude", cwd="/home")
        d = state.to_dict()
        assert d["session_id"] == "s1"
        assert d["agent_type"] == "claude"
        assert "created_at" in d

    def test_from_dict(self):
        d = {
            "session_id": "s1",
            "agent_type": "coco",
            "cwd": "/tmp",
            "message_count": 5,
            "is_active": False,
        }
        state = ACPSessionState.from_dict(d)
        assert state.session_id == "s1"
        assert state.message_count == 5
        assert not state.is_active

    def test_roundtrip(self):
        state = ACPSessionState(session_id="s1", agent_type="claude", cwd="/home", message_count=3, is_active=True)
        d = state.to_dict()
        state2 = ACPSessionState.from_dict(d)
        assert state2.session_id == state.session_id
        assert state2.agent_type == state.agent_type
        assert state2.message_count == state.message_count


class TestPromptResult:
    def test_create(self):
        result = PromptResult(stop_reason="end_turn", text="done")
        assert result.stop_reason == "end_turn"
        assert result.text == "done"
        assert result.tool_calls == []
        assert result.tool_results == []
        assert result.plan is None
        assert result.modified_files == set()

    def test_with_tools(self):
        tc = ToolCallInfo(id="t1", title="Edit", kind="edit", status="completed", locations=["/tmp/f.py"])
        result = PromptResult(
            stop_reason="end_turn",
            tool_calls=[tc],
            modified_files={"/tmp/f.py"},
        )
        assert len(result.tool_calls) == 1
        assert "/tmp/f.py" in result.modified_files

    def test_ingest_history_tracks_files(self):
        result = PromptResult(stop_reason="end_turn")
        result.ingest_history(
            [
                {"kind": "write_file", "data": {"path": "/a.py"}, "ts": time.time()},
                {"kind": "execute", "data": {"command": "echo hi", "exit_code": 0}, "ts": time.time()},
                "bad",
            ]
        )
        assert "/a.py" in result.modified_files
        assert any(e.get("kind") == "execute" for e in result.tool_results)

    def test_render_prompt_result_markdown_contains_sections(self):
        tc = ToolCallInfo(id="t1", title="Read", kind="read", status="completed", locations=["/tmp/a.txt"])
        result = PromptResult(stop_reason="end_turn")
        result.add_text("hello")
        result.add_tool_call(tc)
        md = render_prompt_result_markdown(result)
        assert "PromptResult" in md
        assert "hello" in md
        assert "工具调用" in md
        assert "改动文件" in md


class TestIdleHealthClassificationStatic:
    def test_classify_idle_health_fallback_uses_hooks_and_returns_unknown(self, monkeypatch):
        """ACPSessionManager.classify_idle_health 在输入异常时应调用 Telemetry hook 并回退为 UNKNOWN。"""

        import src.acp.manager as manager_mod
        import src.utils.time_ago as time_ago_mod
        from src.utils.time_ago import IdleHealth

        # 让 SSOT helper 抛出 ValueError，触发回退路径
        def _boom(_bucket):  # type: ignore[unused-argument]
            raise ValueError("bad bucket for static classify test")

        monkeypatch.setattr(time_ago_mod, "classify_idle_health_from_bucket", _boom)

        called: dict[str, object] = {}

        def fake_log_idle_health_classification_fallback(*, bucket, error, context=None) -> None:  # type: ignore[no-untyped-def]
            called["log"] = (bucket, error, context)

        def fake_record_idle_health_fallback_metric(*, error_type: str) -> None:
            called["metric"] = error_type

        # 通过 telemetry 模块级别的私有 hook 名称打桩，验证静态入口会正确转发到
        # Telemetry 层的集中日志与埋点逻辑。
        monkeypatch.setattr(telemetry_mod, "_log_idle_health_classification_fallback", fake_log_idle_health_classification_fallback)
        monkeypatch.setattr(telemetry_mod, "_record_idle_health_fallback_metric", fake_record_idle_health_fallback_metric)

        bucket = {"dummy": True}
        ctx = {"session_key": "s-key-1"}

        health = manager_mod.ACPSessionManager.classify_idle_health(bucket, context=ctx)  # type: ignore[arg-type]

        # 应回退为 UNKNOWN 而不是抛异常
        assert health is IdleHealth.UNKNOWN

        # Telemetry hook 应被调用一次，并收到原始异常和上下文
        assert "log" in called
        assert "metric" in called
        _bucket, err, context = called["log"]  # type: ignore[misc]
        assert isinstance(err, ValueError)
        assert isinstance(context, dict)
        assert context.get("session_key") == "s-key-1"
        assert called["metric"] == "ValueError"


class TestTelemetryPublicExports:
    def test_star_import_exports_telemetry_context_only(self):
        mod = __import__("src.acp.telemetry", fromlist=["*"])
        exported = getattr(mod, "__all__", [])

        # 仅 IdleHealthTelemetryContext 应作为 Telemetry 上下文公开名称出现；
        # IdleHealthContext 作为 [INTERNAL] 实现细节不应通过星号导出暴露给调用方。
        assert "IdleHealthTelemetryContext" in exported
        assert "IdleHealthContext" not in exported

    def test_get_idle_health_service_for_manager_prefers_explicit_instance(self):
        """_get_idle_health_service_for_manager 应优先返回显式传入的 IdleHealthService 实例。"""

        svc = telemetry_mod._IdleHealthService()

        result = telemetry_mod._get_idle_health_service_for_manager(svc)

        assert result is svc

    def test_get_idle_health_service_for_manager_constructs_default_with_telemetry(self):
        """当未显式传入 IdleHealthService 时，应基于给定 telemetry 构造默认实现。"""

        class CustomTelemetry(telemetry_mod._DefaultIdleHealthTelemetry):
            pass

        telemetry = CustomTelemetry()

        result = telemetry_mod._get_idle_health_service_for_manager(None, telemetry=telemetry)

        # 仅通过鸭子类型检查关键方法是否存在，避免对具体实现做过强绑定
        assert hasattr(result, "classify_session_idle_health")

    def test_manager_instance_uses_injected_idle_health_telemetry(self, monkeypatch) -> None:
        """ACPSessionManager 实例应仅依赖 IdleHealthTelemetry 协议方法。"""

        import src.acp.manager as manager_mod
        import src.utils.time_ago as time_ago_mod

        # 让 SSOT helper 抛出 ValueError，从而触发 Telemetry 回退路径。
        def _boom(_bucket):  # type: ignore[unused-argument]
            raise ValueError("bad bucket for instance classify test")

        monkeypatch.setattr(time_ago_mod, "classify_idle_health_from_bucket", _boom)

        class FakeTelemetry:
            def __init__(self) -> None:
                self.logged: list[tuple[dict, object, dict | None]] = []
                self.metrics: list[str] = []

            def log_idle_health_classification_fallback(self, *, bucket, error, context=None):  # type: ignore[no-untyped-def]
                self.logged.append((bucket, error, context))

            def record_idle_health_fallback_metric(self, *, error_type: str) -> None:
                self.metrics.append(error_type)

        telemetry = FakeTelemetry()

        mgr = manager_mod.ACPSessionManager(
            agent_type="coco",
            idle_health_config=telemetry_mod.IdleHealthConfig(
                idle_health_telemetry=telemetry,
            ),
        )

        class DummySession:
            def __init__(self, last_active: float, message_count: int = 3) -> None:
                self.session_id = "sess-instance-1"
                self.last_active = last_active
                self.message_count = message_count

            def to_snapshot(self) -> dict:
                return {"session_id": self.session_id}

            def close(self) -> None:
                return None

        now = time.time()
        dummy = DummySession(last_active=now - 60)

        key = mgr._session_key("chat-inst", "proj-inst", thread_id="thread-inst")
        with mgr._lock:
            mgr._sessions[key] = dummy  # type: ignore[assignment]

        try:
            results = mgr.list_active_sessions()
        finally:
            mgr.cleanup_all()

        # manager 应该通过注入的 telemetry 记录回退日志与指标
        assert telemetry.metrics == ["ValueError"]
        assert len(telemetry.logged) == 1
        bucket, err, ctx = telemetry.logged[0]
        assert isinstance(err, ValueError)
        assert isinstance(ctx, dict)
        assert ctx.get("session_id") == "sess-instance-1"
        assert ctx.get("chat_id") == "chat-inst"
        assert ctx.get("project_id") == "proj-inst"
        assert ctx.get("thread_id") == "thread-inst"

        # list_active_sessions 返回的 idle_health 也应是 UNKNOWN（经由 Telemetry 回退逻辑）。
        from src.utils.time_ago import IdleHealth

        assert len(results) == 1
        info = results[0]
        assert isinstance(info["idle_health"], IdleHealth)
        assert info["idle_health"] is IdleHealth.UNKNOWN
