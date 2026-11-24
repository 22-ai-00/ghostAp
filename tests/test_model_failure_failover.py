import logging

import src.agent_session.factory as session_factory
import src.agent_session.model_diagnostics as model_diagnostics
import src.agent_session.wrappers as session_wrappers
from src.agent_session.factory import create_engine_session
from src.agent_session.wrappers import ModelFailureAwareSession


def test_e2e_model_failure_need_compaction_then_loop_then_failover(monkeypatch, caplog):
    """端到端回归夹具：need compaction -> loop -> failover 到 gpt-5.1 后成功。

    约束：
    - 不启动真实子进程
    - 通过 monkeypatch 构造 create_engine_session 产生的包装链路
    """
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

    monkeypatch.setattr(session_factory, "get_settings", lambda: _S())
    monkeypatch.setattr(session_wrappers, "get_settings", lambda: _S())

    # Base session that simulates model errors across calls
    class _Base:
        def __init__(self, model: str):
            self.session_id = "sid"
            self.created_at = 0.0
            self.last_active = 0.0
            self.message_count = 0
            self.last_query = ""
            self._agent_type = "coco"
            self._cwd = "/tmp"
            self._agent_cmd = "coco"
            self._agent_args = ["acp", "serve", "-c", f"model.name={model}"]
            self.calls = 0

        def describe_agent(self):
            return "dummy"

        def start(self, startup_timeout: float = 60):
            return self.session_id


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
            cur_model = model_diagnostics._extract_model_from_agent_args(self._agent_args)
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
        model = model_diagnostics._extract_model_from_agent_args(
            list(getattr(sess, "_agent_args", []) or [])
        )
        return _Base(model)

    # Patch SyncACPSession used by failover rebuild
    def _fake_sync_acp_session(**kw):
        model = model_diagnostics._extract_model_from_agent_args(
            list(kw.get("agent_args") or [])
        )
        return _Base(model)

    monkeypatch.setattr(session_wrappers, "SyncACPSession", _fake_sync_acp_session)

    # Create session through factory (ensures wrapper chain is applied)
    sess = create_engine_session(agent_type="coco", cwd="/tmp", model_name="gpt-5.2")
    assert isinstance(sess, ModelFailureAwareSession)
    # inject compaction action for deterministic behavior
    sess._compaction_action = _compaction_action

    r = sess.send_prompt("hi")
    assert getattr(r, "text", "") == "ok"

    joined = "\n".join([x.getMessage() for x in caplog.records]).lower()
    assert "action=compaction" in joined
    assert "action=failover" in joined
    assert "to_model=gpt-5.1" in joined


def test_parse_failover_map_preserves_dsh_json_arrays_and_executes_failover(monkeypatch, tmp_path):
    """_parse_failover_map handles DSH JSON-array selections without splitting internal commas."""
    settings = type(
        "_S",
        (),
        {
            "acp_startup_timeout": 2,
            "model_failure_compaction_enabled": True,
            "model_failure_compaction_loop_window_s": 999.0,
            "model_failure_compaction_loop_max": 2,
            "model_failure_failover_map": (
                '["deepseek","deepseek-reasoner","high"]:["deepseek","deepseek-chat","low"], '
                '["openrouter","qwen:32b","medium"]:["openrouter","gpt-5","low"]; '
                "gpt-5.2:gpt-5.1"
            ),
        },
    )()
    monkeypatch.setattr(session_wrappers, "get_settings", lambda: settings)

    class _DshBase:
        def __init__(self, model_name: str):
            self._agent_type = "dsh"
            self._cwd = str(tmp_path)
            self._agent_cmd = "dsh"
            self._agent_args = ["--profile", "acp"]
            self._model_name = model_name
            self._explicit_env = {"DSH_ENV": "1"}
            self._capture_full_tool_content = False

        def start(self, startup_timeout: float = 20):
            return "sid-dsh"

        def close(self):
            return None

        def send_prompt(self, _text, **_kwargs):
            if self._model_name == '["deepseek","deepseek-reasoner","high"]':
                raise RuntimeError("loop detected")
            return type("R", (), {"stop_reason": "end_turn", "text": f"ok:{self._model_name}"})()

    created_models: list[str | None] = []

    def _fake_sync_acp(**kw):
        created_models.append(kw.get("model_name"))
        return _DshBase(kw.get("model_name") or "")

    monkeypatch.setattr(session_wrappers, "SyncACPSession", _fake_sync_acp)

    wrapper = ModelFailureAwareSession(_DshBase('["deepseek","deepseek-reasoner","high"]'))
    assert wrapper._parse_failover_map() == {
        '["deepseek","deepseek-reasoner","high"]': '["deepseek","deepseek-chat","low"]',
        '["openrouter","qwen:32b","medium"]': '["openrouter","gpt-5","low"]',
        "gpt-5.2": "gpt-5.1",
    }

    result = wrapper.send_prompt("hello")
    assert result.text == 'ok:["deepseek","deepseek-chat","low"]'
    assert created_models == ['["deepseek","deepseek-chat","low"]']

    # JSON dict format is also accepted cleanly
    settings.model_failure_failover_map = (
        '{"[\\"deepseek\\",\\"deepseek-reasoner\\",\\"high\\"]": '
        '"[\\"deepseek\\",\\"deepseek-chat\\",\\"low\\"]"}'
    )
    assert wrapper._parse_failover_map() == {
        '["deepseek","deepseek-reasoner","high"]': '["deepseek","deepseek-chat","low"]',
    }


def test_codex_config_c_failover_strips_stale_effort_only_when_target_lacks_effort(
    monkeypatch, tmp_path
) -> None:
    """B068: failover to effort-less codex model strips -c model_reasoning_effort=...; explicit target effort updates it."""
    captured_args: list[list[str]] = []

    class _CodexConfigCBase:
        def __init__(self, agent_args: list[str], model_name: str):
            self._agent_type = "codex"
            self._cwd = str(tmp_path)
            self._agent_cmd = "codex"
            self._agent_args = list(agent_args)
            self._model_name = model_name
            self._capture_full_tool_content = False

        def start(self, startup_timeout: float = 20):
            return "sid-codex"

        def close(self):
            return None

    def _fake_sync_acp(**kw):
        args = list(kw.get("agent_args") or [])
        captured_args.append(args)
        return _CodexConfigCBase(args, kw.get("model_name") or "")

    monkeypatch.setattr(session_wrappers, "SyncACPSession", _fake_sync_acp)

    initial_args = [
        "acp",
        "serve",
        "-c",
        "model.name=gpt-5.4",
        "-c",
        'model_reasoning_effort="xhigh"',
    ]

    # 1. Failover to model without explicit effort suffix -> strips stale effort flag
    w_plain = ModelFailureAwareSession(_CodexConfigCBase(initial_args, "gpt-5.4/xhigh"))
    assert w_plain._do_failover(from_model="gpt-5.4/xhigh", to_model="gpt-5.3-codex") is True
    assert captured_args[-1] == ["acp", "serve", "-c", "model.name=gpt-5.3-codex"]

    # 2. Failover to model with explicit effort suffix -> updates -c model_reasoning_effort=... to target effort
    w_effort = ModelFailureAwareSession(_CodexConfigCBase(initial_args, "gpt-5.4/xhigh"))
    assert w_effort._do_failover(from_model="gpt-5.4/xhigh", to_model="gpt-5.3-codex/high") is True
    assert captured_args[-1] == [
        "acp",
        "serve",
        "-c",
        "model.name=gpt-5.3-codex",
        "-c",
        'model_reasoning_effort="high"',
    ]

    # 3. Failover to default model -> strips model & effort; default/medium updates effort
    w_default = ModelFailureAwareSession(_CodexConfigCBase(initial_args, "gpt-5.4/xhigh"))
    assert w_default._do_failover(from_model="gpt-5.4/xhigh", to_model="default") is True
    assert captured_args[-1] == ["acp", "serve"]

    w_default_effort = ModelFailureAwareSession(_CodexConfigCBase(initial_args, "gpt-5.4/xhigh"))
    assert w_default_effort._do_failover(from_model="gpt-5.4/xhigh", to_model="default/medium") is True
    assert captured_args[-1] == ["acp", "serve", "-c", 'model_reasoning_effort="medium"']


def test_claude_failover_splits_model_and_updates_or_strips_effort_args(
    monkeypatch, tmp_path
) -> None:
    """Claude failover splits opus/high into --model opus + --effort high and strips stale --effort."""
    captured: list[tuple[list[str], str | None]] = []

    class _ClaudeBase:
        def __init__(self, agent_args: list[str], model_name: str):
            self._agent_type = "claude"
            self._cwd = str(tmp_path)
            self._agent_cmd = "claude"
            self._agent_args = list(agent_args)
            self._model_name = model_name
            self._capture_full_tool_content = False

        def start(self, startup_timeout: float = 20):
            return "sid-claude"

        def close(self):
            return None

    def _fake_sync_acp(**kw):
        args = list(kw.get("agent_args") or [])
        captured.append((args, kw.get("model_name")))
        return _ClaudeBase(args, kw.get("model_name") or "")

    monkeypatch.setattr(session_wrappers, "SyncACPSession", _fake_sync_acp)

    initial = ["-p", "--model", "sonnet", "--effort", "low", "--", "hello"]
    w1 = ModelFailureAwareSession(_ClaudeBase(initial, "sonnet/low"))
    assert w1._do_failover(from_model="sonnet/low", to_model="opus/high") is True
    assert captured[-1][0] == ["-p", "--model", "opus", "--effort", "high", "--", "hello"]
    assert captured[-1][1] == "opus/high"

    # Target without effort strips stale --effort low
    w2 = ModelFailureAwareSession(_ClaudeBase(initial, "sonnet/low"))
    assert w2._do_failover(from_model="sonnet/low", to_model="opus") is True
    assert captured[-1][0] == ["-p", "--model", "opus", "--", "hello"]

    # Target default/max strips --model and updates --effort max
    w3 = ModelFailureAwareSession(_ClaudeBase(initial, "sonnet/low"))
    assert w3._do_failover(from_model="sonnet/low", to_model="default/max") is True
    assert captured[-1][0] == ["-p", "--effort", "max", "--", "hello"]

    # Real SyncClaudeCLISession rebuilds via SyncClaudeCLISession without calling SyncACPSession
    from src.agent_session.claude_cli import ClaudeCLIConfig, SyncClaudeCLISession

    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    cli_base = SyncClaudeCLISession(
        cwd=str(tmp_path),
        config=ClaudeCLIConfig(command="claude"),
        model_name="sonnet/low",
    )
    cli_base.start()
    monkeypatch.setattr(
        session_wrappers,
        "SyncACPSession",
        lambda **_kw: (_ for _ in ()).throw(AssertionError("CLI failover must not call SyncACPSession")),
    )
    w_cli = ModelFailureAwareSession(cli_base)
    assert w_cli._do_failover(from_model="sonnet/low", to_model="opus[1m]/high") is True
    rebuilt_cli = w_cli._inner
    assert isinstance(rebuilt_cli, SyncClaudeCLISession)
    argv = rebuilt_cli._build_args("hi", False)
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "high"


def test_effort_backends_failover_and_attach_preserve_selection(
    monkeypatch, tmp_path
) -> None:
    """Traex, DSH, and AGY preserve compound model/profile/effort selections across failover."""
    captured: list[dict[str, object]] = []

    class _AcpBase:
        def __init__(self, agent_type: str, agent_args: list[str], model_name: str):
            self._agent_type = agent_type
            self._cwd = str(tmp_path)
            self._agent_cmd = agent_type
            self._agent_args = list(agent_args)
            self._model_name = model_name
            self._capture_full_tool_content = False

        def start(self, startup_timeout: float = 20):
            return f"sid-{self._agent_type}"

        def close(self):
            return None

    def _fake_sync_acp(**kw):
        captured.append(dict(kw))
        return _AcpBase(kw["agent_type"], list(kw.get("agent_args") or []), kw.get("model_name") or "")

    monkeypatch.setattr(session_wrappers, "SyncACPSession", _fake_sync_acp)
    monkeypatch.setattr(
        "src.acp.providers.TraexACPProvider._load_slug_map",
        lambda _self: {"c_o_new_thinking": "Test-O-New-Thinking"},
    )

    for tool, args, from_m, to_m, expected_model_arg in (
        ("traex", ["acp", "serve", "-c", 'model="Test-O-Old"'], "c_o_old/standard/low", "c_o_new_thinking/max/xhigh", 'model="Test-O-New-Thinking"'),
        ("dsh", ["--profile", "acp"], '["deepseek","deepseek-chat","low"]', '["deepseek","deepseek-reasoner","high"]', "--profile"),
        ("agy", ["acp"], "gemini-4-hydrogen/low", "gemini-4-neon/xhigh", "acp"),
    ):
        w = ModelFailureAwareSession(_AcpBase(tool, args, from_m))
        assert w._do_failover(from_model=from_m, to_model=to_m) is True
        assert captured[-1]["model_name"] == to_m
        assert expected_model_arg in captured[-1]["agent_args"]


def test_agy_cli_to_non_orchestrator_failover_switches_to_sync_acp_session(
    monkeypatch, tmp_path
) -> None:
    """REAL_BUG-12: SyncAGYCLISession failover to non-orchestrator model switches cleanly to SyncACPSession."""
    from src.acp.sync_adapter import SyncACPSession
    from src.agent_session.agy_cli import AGYCLIConfig, SyncAGYCLISession
    from src.agent_session.wrappers import RateLimitAwareSession

    closed_calls: list[str] = []
    started_models: list[str | None] = []

    cli_base = SyncAGYCLISession(
        cwd=str(tmp_path),
        config=AGYCLIConfig(command="agy", show_thinking=False),
        model_name="gemini-pro-orchestrator-neon/high/low",
    )
    monkeypatch.setattr(cli_base, "close", lambda: closed_calls.append("cli_closed"))

    monkeypatch.setattr(
        session_wrappers,
        "resolve_agent_spec",
        lambda agent_type, model_name=None: ("agy", ["acp"]),
    )

    class _FakeACP(SyncACPSession):
        def __init__(self, **kw):
            super().__init__(**kw)

        def start(self, startup_timeout: float = 60):
            started_models.append(self._model_name)
            return "sid-agy-acp"

        def close(self):
            return None

    monkeypatch.setattr(session_wrappers, "SyncACPSession", _FakeACP)

    wrapper = ModelFailureAwareSession(RateLimitAwareSession(cli_base))
    assert (
        wrapper._do_failover(
            from_model="gemini-pro-orchestrator-neon/high/low",
            to_model="gemini-4-neon/high",
        )
        is True
    )
    assert closed_calls == ["cli_closed"]
    assert isinstance(wrapper._inner, RateLimitAwareSession)
    rebuilt = wrapper._inner._inner
    assert isinstance(rebuilt, _FakeACP)
    assert rebuilt._agent_type == "agy"
    assert rebuilt._agent_cmd == "agy"
    assert rebuilt._agent_args == ["acp"]
    assert rebuilt._model_name == "gemini-4-neon/high"
    assert wrapper._agent_type == "agy"
    assert started_models == ["gemini-4-neon/high"]

    # Provider resolution failure returns False cleanly without closing the active session
    cli_base_fail = SyncAGYCLISession(
        cwd=str(tmp_path),
        config=AGYCLIConfig(command="agy"),
        model_name="gemini-pro-orchestrator-neon/high/low",
    )
    fail_closed: list[str] = []
    monkeypatch.setattr(cli_base_fail, "close", lambda: fail_closed.append("closed"))
    monkeypatch.setattr(
        session_wrappers,
        "resolve_agent_spec",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("agy ACP unavailable")),
    )
    wrapper_fail = ModelFailureAwareSession(cli_base_fail)
    assert (
        wrapper_fail._do_failover(
            from_model="gemini-pro-orchestrator-neon/high/low",
            to_model="gemini-4-neon/high",
        )
        is False
    )
    assert fail_closed == []
    assert wrapper_fail._inner is cli_base_fail


def test_agy_cli_to_orchestrator_failover_rebuilds_agy_cli_session(
    monkeypatch, tmp_path
) -> None:
    """REAL_BUG-12: SyncAGYCLISession failover to another orchestrator model rebuilds SyncAGYCLISession cleanly."""
    from src.agent_session.agy_cli import AGYCLIConfig, SyncAGYCLISession

    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        session_wrappers,
        "SyncACPSession",
        lambda **_kw: (_ for _ in ()).throw(
            AssertionError("AGY orchestrator failover must not call SyncACPSession")
        ),
    )

    closed_calls: list[str] = []
    cli_base = SyncAGYCLISession(
        cwd=str(tmp_path),
        config=AGYCLIConfig(command="agy", show_thinking=False),
        model_name="gemini-pro-orchestrator-neon/high/low",
    )
    cli_base.start()
    monkeypatch.setattr(cli_base, "close", lambda: closed_calls.append("cli_closed"))

    wrapper = ModelFailureAwareSession(cli_base)
    assert (
        wrapper._do_failover(
            from_model="gemini-pro-orchestrator-neon/high/low",
            to_model="gemini-pro-orchestrator/high/high",
        )
        is True
    )
    assert closed_calls == ["cli_closed"]
    rebuilt = wrapper._inner
    assert isinstance(rebuilt, SyncAGYCLISession)
    assert rebuilt is not cli_base
    assert rebuilt._agent_type == "agy"
    assert rebuilt._cfg.show_thinking is False
    assert rebuilt._model_name == "gemini-pro-orchestrator/high/high"
    argv = rebuilt._build_args("hello", False)
    assert argv[argv.index("--model") + 1] == "gemini-pro-orchestrator"
    assert "--settings" in argv


def test_agy_acp_to_orchestrator_failover_switches_to_agy_cli_session(
    monkeypatch, tmp_path
) -> None:
    """REAL_BUG-12: SyncACPSession(agent_type='agy') failover to orchestrator switches to SyncAGYCLISession."""
    from src.acp.sync_adapter import SyncACPSession
    from src.agent_session.agy_cli import SyncAGYCLISession

    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    acp_base = SyncACPSession(
        agent_type="agy",
        cwd=str(tmp_path),
        agent_cmd="agy",
        agent_args=["acp"],
        model_name="gemini-4-neon/high",
    )
    closed_calls: list[str] = []
    monkeypatch.setattr(acp_base, "close", lambda: closed_calls.append("acp_closed"))
    monkeypatch.setattr(
        session_wrappers,
        "SyncACPSession",
        lambda **_kw: (_ for _ in ()).throw(
            AssertionError("AGY orchestrator failover must switch to SyncAGYCLISession")
        ),
    )

    wrapper = ModelFailureAwareSession(acp_base)
    assert (
        wrapper._do_failover(
            from_model="gemini-4-neon/high",
            to_model="gemini-pro-orchestrator-neon/high/low",
        )
        is True
    )
    assert closed_calls == ["acp_closed"]
    rebuilt = wrapper._inner
    assert isinstance(rebuilt, SyncAGYCLISession)
    assert rebuilt._agent_type == "agy"
    assert wrapper._agent_type == "agy"
    assert rebuilt._model_name == "gemini-pro-orchestrator-neon/high/low"
    argv = rebuilt._build_args("hello", False)
    assert argv[argv.index("--model") + 1] == "gemini-pro-orchestrator-neon"

    # Missing CLI binary returns False cleanly on SyncAGYCLISession.start() failure
    monkeypatch.setattr("shutil.which", lambda _cmd: None)
    acp_base_missing = SyncACPSession(
        agent_type="agy",
        cwd=str(tmp_path),
        agent_cmd="agy",
        agent_args=["acp"],
        model_name="gemini-4-neon/high",
    )
    monkeypatch.setattr(acp_base_missing, "close", lambda: None)
    wrapper_missing = ModelFailureAwareSession(acp_base_missing)
    assert (
        wrapper_missing._do_failover(
            from_model="gemini-4-neon/high",
            to_model="gemini-pro-orchestrator-neon/high/low",
        )
        is False
    )


def test_codex_w_empty_model_failover_to_default_effort_succeeds(
    monkeypatch, tmp_path
) -> None:
    """REAL_BUG-12: empty-model SyncACPSession(agent_type='codex_w') failover to 'default/high' succeeds."""
    from src.acp.sync_adapter import SyncACPSession

    captured: list[dict[str, object]] = []

    class _FakeCodexWACP(SyncACPSession):
        def __init__(self, **kw):
            captured.append(dict(kw))
            super().__init__(**kw)

        def start(self, startup_timeout: float = 60):
            return "sid-codex-w"

        def close(self):
            return None

    monkeypatch.setattr(session_wrappers, "SyncACPSession", _FakeCodexWACP)

    base = _FakeCodexWACP(
        agent_type="codex_w",
        cwd=str(tmp_path),
        agent_cmd="codex-w",
        agent_args=["acp", "serve"],
        model_name="",
    )
    captured.clear()

    wrapper = ModelFailureAwareSession(base)
    assert wrapper._do_failover(from_model="default", to_model="default/high") is True
    assert len(captured) == 1
    assert captured[0]["agent_type"] == "codex_w"
    assert captured[0]["model_name"] == "default/high"
    assert captured[0]["agent_args"] == ["acp", "serve"]


