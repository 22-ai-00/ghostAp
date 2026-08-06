import logging


def test_e2e_model_failure_need_compaction_then_loop_then_failover(monkeypatch, caplog):
    """端到端回归夹具：need compaction -> loop -> failover 到 gpt-5.1 后成功。

    约束：
    - 不启动真实子进程
    - 通过 monkeypatch 构造 create_engine_session 产生的包装链路
    """
    import src.agent_session as agent_session

    caplog.set_level(logging.WARNING)

    # Settings: force deterministic loop threshold
    class _S:
        rate_limit_retry_enabled = False
        rate_limit_max_wait = 300
        rate_limit_base_wait = 1
        rate_limit_max_retries = 1
        acp_startup_timeout = 1
        model_failure_compaction_enabled = True
        model_failure_compaction_loop_window_s = 999.0
        model_failure_compaction_loop_max = 2  # second compaction event triggers loop
        model_failure_failover_map = "gpt-5.2:gpt-5.1"

    monkeypatch.setattr(agent_session.factory, "get_settings", lambda: _S())
    monkeypatch.setattr(agent_session.wrappers, "get_settings", lambda: _S())

    # Base session that simulates model errors across calls
    class _Base:
        def __init__(self, model: str):
            self.session_id = "sid"
            self.created_at = 0.0
            self.last_active = 0.0
            self.message_count = 0
            self.last_query = ""
            self.is_resumed = False
            self._agent_type = "coco"
            self._cwd = "/tmp"
            self._agent_cmd = "coco"
            self._agent_args = ["acp", "serve", "-c", f"model.name={model}"]
            self.calls = 0

        def describe_agent(self):
            return "dummy"

        def start(self, startup_timeout: float = 60):
            return self.session_id

        def load_session(self, session_id: str, timeout: float = 60):
            del timeout
            return None

        def load_local_history(self, session_id=None, limit: int = 200):
            return []

        def cancel(self):
            return None

        def close(self):
            return None

        def to_snapshot(self):
            return {}

        def get_session_info(self):
            return ""

        def is_server_running(self):
            return True

        def is_server_healthy(self, healthcheck_timeout: float = 2.0):
            return True

        def send_prompt(self, text: str, on_event=None, timeout=None):
            self.calls += 1
            # model gpt-5.2: first call -> need compaction; second call -> need compaction again triggers loop -> failover
            cur_model = agent_session._extract_model_from_agent_args(self._agent_args)
            if cur_model == "gpt-5.2":
                if self.calls == 1:
                    raise RuntimeError("Model failed: model 'gpt-5.2': receive message: need compaction")
                raise RuntimeError("Model failed: model 'gpt-5.2': receive message: need compaction")

            # model gpt-5.1: success
            return type("R", (), {"stop_reason": "end_turn", "text": "ok"})()

    # start_session_with_retry should return base session with initial model
    monkeypatch.setattr(
        "src.acp.sync_adapter.start_session_with_retry",
        lambda **kw: _Base(kw.get("model_name") or "gpt-5.2"),
    )

    # Compaction action: rebuild a new base session with same model
    def _compaction_action(sess):
        model = agent_session._extract_model_from_agent_args(list(getattr(sess, "_agent_args", []) or []))
        return _Base(model)

    # Patch SyncACPSession used by failover rebuild
    def _fake_sync_acp_session(**kw):
        model = agent_session._extract_model_from_agent_args(list(kw.get("agent_args") or []))
        return _Base(model)

    monkeypatch.setattr(agent_session.wrappers, "SyncACPSession", _fake_sync_acp_session)

    # Create session through factory (ensures wrapper chain is applied)
    sess = agent_session.create_engine_session(agent_type="coco", cwd="/tmp", model_name="gpt-5.2")
    assert isinstance(sess, agent_session.ModelFailureAwareSession)
    # inject compaction action for deterministic behavior
    sess._compaction_action = _compaction_action

    r = sess.send_prompt("hi")
    assert getattr(r, "text", "") == "ok"

    joined = "\n".join([x.getMessage() for x in caplog.records]).lower()
    assert "action=compaction" in joined
    assert "action=failover" in joined
    assert "to_model=gpt-5.1" in joined
