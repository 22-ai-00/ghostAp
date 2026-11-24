"""Behavioral tests for Workflow tool + model + effort identity, discovery, and passthrough."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.acp.helper import (
    _cache_lock,
    _probe_cache,
    get_cached_acp_models,
    invalidate_acp_model_cache,
)
from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.acp.providers import CodexACPProvider, _ProviderConfig
from src.acp.sync_adapter import SyncACPSession
from src.agent_session.model_diagnostics import (
    _apply_compaction_once,
    _extract_model_from_agent_args,
    _replace_model_in_agent_args,
)
from src.agent_session.wrappers import ModelFailureAwareSession
from src.spec_engine.models import ReviewAgentBinding
from src.workflow_engine.agent_pool import (
    WorkflowAgentBinding,
    freeze_agent_pool,
    resolve_effective_model_name,
    select_auto_orchestrator,
)
from src.workflow_engine.engine import WorkflowEngine, WorkflowEngineCallbacks
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.journal import WorkflowJournal
from src.workflow_engine.models import (
    AgentCallParams,
    AgentCallResult,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.run_spec import WorkflowRunSpec
from src.workflow_engine.state_manager import WorkflowStateManager
from src.workflow_engine.tool_registry import (
    EFFORT_CAPABLE_TOOLS,
    MODEL_ONLY_TOOLS,
    ToolCapability,
    get_tool_capabilities,
    invalidate_cache,
    validate_agent_binding_spec,
)


@pytest.fixture(autouse=True)
def _clean_tool_caches(tmp_path):
    invalidate_cache()
    with _cache_lock:
        _probe_cache.pop(("codex", str(tmp_path)), None)
    yield
    invalidate_cache()
    with _cache_lock:
        for key in [k for k in _probe_cache if k[0] in (*EFFORT_CAPABLE_TOOLS, *MODEL_ONLY_TOOLS)]:
            _probe_cache.pop(key, None)
    for tool in (*EFFORT_CAPABLE_TOOLS, *MODEL_ONLY_TOOLS):
        invalidate_acp_model_cache(tool)


def test_get_tool_capabilities_is_non_blocking_and_distinguishes_effort_backends(tmp_path) -> None:
    """Capability discovery must not trigger sync ACP probes on cache miss."""
    seen_sync_probes: list[bool] = []

    def fake_avail(name: str, **kwargs) -> bool:
        seen_sync_probes.append(bool(kwargs.get("allow_sync_probe")))
        if kwargs.get("allow_sync_probe"):
            raise AssertionError("get_tool_capabilities invoked blocking sync probe")
        return name == "codex"

    def fake_which(cmd: str) -> str | None:
        return f"/bin/{cmd}" if cmd in {"claude-w", "codex-w", "coco"} else None

    with (
        patch("src.acp.helper.fetch_acp_models", side_effect=AssertionError("must not block")),
        patch("src.acp.helper.is_programming_tool_available", side_effect=fake_avail),
        patch("src.acp.providers.get_providers", return_value={}),
        patch("src.acp.traex_selection.load_traex_model_metadata", return_value=()),
        patch("shutil.which", side_effect=fake_which),
    ):
        caps = get_tool_capabilities(force_refresh=True, cwd=str(tmp_path))

    assert set(caps) == EFFORT_CAPABLE_TOOLS | MODEL_ONLY_TOOLS
    assert seen_sync_probes and all(flag is False for flag in seen_sync_probes)

    for tool in EFFORT_CAPABLE_TOOLS:
        assert caps[tool].supports_effort is True
        assert caps[tool].supports_model is True
    for tool in MODEL_ONLY_TOOLS:
        assert caps[tool].supports_effort is False
        assert caps[tool].supports_model is True

    # CLI bridges resolve availability via cli_command_for_agent (`claude-w`, `codex-w`)
    assert caps["claude_w"].available is True
    assert caps["codex_w"].available is True
    assert caps["claude"].available is False
    assert caps["coco"].available is True
    assert caps["codex"].available is True

    # Returned dict copy & frozen ToolCapability protect cache from external mutation
    caps["MUTATED"] = caps["coco"]
    with pytest.raises(TypeError):
        caps["claude_w"].efforts_by_model["default"] = ("low",)  # type: ignore[index]
    fresh = get_tool_capabilities(cwd=str(tmp_path))
    assert "MUTATED" not in fresh


def test_get_tool_capabilities_reads_warm_acp_cache_and_traex_metadata(tmp_path) -> None:
    """Warm ACP probe cache and Traex local metadata populate models and efforts without spawning."""
    cwd = str(tmp_path)
    traex_cache = tmp_path / "models_cache.json"
    traex_cache.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "Test-O-New-Thinking",
                        "config_name": "c_o_new_thinking",
                        "business_metadata": {
                            "variants": {
                                "standard_supported_reasoning_levels": [
                                    {"effort": "low"},
                                    {"effort": "high"},
                                ],
                                "standard_default_reasoning_level": "low",
                                "max_key": "c_o_new_thinking_max",
                                "max_supported_reasoning_levels": [{"effort": "xhigh"}],
                                "max_default_reasoning_level": "xhigh",
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    codex_options = [
        ACPModelOption(
            name="gpt-5.4",
            is_default=True,
            reasoning_efforts=("low", "medium", "xhigh"),
            default_reasoning_effort="medium",
            selection_variants=(
                ACPModelSelectionVariant("gpt-5.4/low", "gpt-5.4", effort="low"),
                ACPModelSelectionVariant("gpt-5.4/medium", "gpt-5.4", effort="medium", is_default=True),
                ACPModelSelectionVariant("gpt-5.4/xhigh", "gpt-5.4", effort="xhigh"),
            ),
        )
    ]
    with _cache_lock:
        _probe_cache[("codex", cwd)] = (time.time(), codex_options)

    from src.acp.traex_selection import load_traex_model_metadata

    real_traex_meta = load_traex_model_metadata(traex_cache)
    with (
        patch("src.acp.helper.is_programming_tool_available", return_value=True),
        patch("src.acp.providers.get_providers", return_value={}),
        patch("src.acp.traex_selection.load_traex_model_metadata", return_value=real_traex_meta),
        patch("shutil.which", return_value="/bin/tool"),
    ):
        assert [opt.name for opt in get_cached_acp_models("codex", cwd)] == ["gpt-5.4"]
        caps = get_tool_capabilities(force_refresh=True, cwd=cwd)

    assert caps["codex"].models == ("gpt-5.4",)
    assert caps["codex"].efforts_by_model["gpt-5.4"] == ("low", "medium", "xhigh")
    assert caps["codex"].default_effort == "medium"

    assert caps["traex"].models == ("c_o_new_thinking",)
    assert caps["traex"].profiles_by_model["c_o_new_thinking"] == ("standard", "max")
    assert caps["traex"].efforts_by_model["c_o_new_thinking"] == ("low", "high", "xhigh")
    assert caps["traex"].default_effort == "low"


def test_validate_agent_binding_spec_enforces_backend_capabilities(tmp_path) -> None:
    """Validation rejects unknown/unavailable tools, unsupported effort/profile, and bad values."""
    caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "xhigh")},
            default_effort="medium",
        ),
        "claude": ToolCapability(
            tool_name="claude",
            display_name="Claude",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("default",),
            efforts_by_model={"default": ("low", "medium", "high", "xhigh", "max")},
        ),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
        ),
        "grok": ToolCapability(
            tool_name="grok",
            display_name="Grok",
            available=False,
            supports_model=True,
            supports_effort=False,
        ),
    }

    ok, err = validate_agent_binding_spec("", capabilities=caps)
    assert ok is False and err is not None

    ok, err = validate_agent_binding_spec("nonexistent", capabilities=caps)
    assert ok is False and "未知" in (err or "")

    ok, err = validate_agent_binding_spec("grok", capabilities=caps)
    assert ok is False and "不可用" in (err or "")

    # Model-only backend rejects explicit effort or profile, but keeps slash model IDs
    ok, err = validate_agent_binding_spec("coco", model_name="anthropic/claude-sonnet", effort="high", capabilities=caps)
    assert ok is False and "effort" in (err or "")

    ok, err = validate_agent_binding_spec("coco", model_name="anthropic/claude-sonnet", profile="max", capabilities=caps)
    assert ok is False and "profile" in (err or "")

    ok, err = validate_agent_binding_spec("coco", model_name="anthropic/claude-sonnet", capabilities=caps)
    assert ok is True and err is None

    # Effort-capable backends check model + effort + profile matrices
    ok, err = validate_agent_binding_spec("codex", model_name="gpt-5.4", effort="ultra", capabilities=caps)
    assert ok is False and "effort" in (err or "")

    ok, err = validate_agent_binding_spec("codex", model_name="unknown-model", effort="low", capabilities=caps)
    assert ok is False and "模型" in (err or "")

    ok, err = validate_agent_binding_spec("codex", model_name=None, effort="high", capabilities=caps)
    assert ok is True and err is None

    ok, err = validate_agent_binding_spec("claude", model_name="claude-sonnet-4-5", effort="ultra", capabilities=caps)
    assert ok is False and "effort" in (err or "")

    ok, err = validate_agent_binding_spec("claude", model_name="claude-sonnet-4-5", effort="max", capabilities=caps)
    assert ok is True and err is None


def test_same_tool_and_model_with_distinct_effort_coexist_and_rank_cleanly() -> None:
    """Two bindings with the same tool + model but different efforts coexist in a pool."""
    b_low = WorkflowAgentBinding(
        agent_id="A1",
        tool_name="codex",
        model_name="gpt-5.4",
        display_name="Codex Low",
        effort=" low ",
        role=" Worker ",
        roles=("worker", "Scout", "worker"),
    )
    b_xhigh = WorkflowAgentBinding(
        agent_id="A2",
        tool_name="codex",
        model_name="gpt-5.4",
        display_name="Codex XHigh",
        effort="XHIGH",
        role="reviewer",
    )

    assert b_low.effort == "low"
    assert b_low.role == "worker"
    assert b_low.roles == ("worker", "scout")
    assert b_low.binding_key == ("codex", "gpt-5.4", "", "low")
    assert b_xhigh.binding_key == ("codex", "gpt-5.4", "", "xhigh")

    pool = freeze_agent_pool([b_low, b_xhigh])
    assert len(pool) == 2

    round_tripped = WorkflowAgentBinding.from_dict(b_low.to_dict())
    assert round_tripped == b_low

    # Exact 4-tuple duplicates still fail with expected message pattern
    dup = WorkflowAgentBinding(
        agent_id="A3",
        tool_name="codex",
        model_name="gpt-5.4",
        display_name="Codex Low Dup",
        effort="low",
    )
    with pytest.raises(ValueError, match="duplicate.*tool.*model"):
        freeze_agent_pool([b_low, dup])

    # 2-tuple recommendations still rank against 4-tuple bindings
    selected = select_auto_orchestrator(
        [b_low, b_xhigh],
        recommendations=[{"tool_name": "codex", "model_name": "gpt-5.4", "effort": "xhigh"}],
    )
    assert selected.agent_id == "A2"

    selected_2tuple = select_auto_orchestrator(
        [
            WorkflowAgentBinding("A1", "coco", "m1", "Coco"),
            b_xhigh,
        ],
        recommendations=[{"tool_name": "codex", "model_name": "gpt-5.4"}],
    )
    assert selected_2tuple.agent_id == "A2"


@pytest.mark.parametrize(
    ("tool", "model", "profile", "effort", "expected"),
    [
        ("codex", "gpt-5.4", None, "xhigh", "gpt-5.4/xhigh"),
        ("codex", None, None, "high", "default/high"),
        ("codex_w", None, None, "xhigh", "default/xhigh"),
        ("claude", "claude-sonnet-4-5[1m]", None, "max", "claude-sonnet-4-5[1m]/max"),
        ("claude_w", None, None, "high", "default/high"),
        ("agy", "gemini-4-neon", None, "high", "gemini-4-neon/high"),
        (
            "agy",
            "gemini-pro-orchestrator-neon",
            "high",
            "low",
            "gemini-pro-orchestrator-neon/high/low",
        ),
        (
            "agy",
            "gemini-pro-orchestrator-neon",
            None,
            "high",
            "gemini-pro-orchestrator-neon/high/high",
        ),
        ("traex", "c_o_new_thinking", "max", "xhigh", "c_o_new_thinking/max/xhigh"),
        ("traex", "c_o_new_thinking", "standard", None, "c_o_new_thinking"),
        ("dsh", "deepseek/deepseek-reasoner", None, "high", '["deepseek","deepseek-reasoner","high"]'),
        ("dsh", '["deepseek","deepseek-reasoner"]', None, "off", '["deepseek","deepseek-reasoner","off"]'),
        ("coco", "anthropic/claude-3.5-sonnet", None, None, "anthropic/claude-3.5-sonnet"),
        ("aiden", "org/model", None, None, "org/model"),
        ("gemini", "gemini-2.5-pro", None, None, "gemini-2.5-pro"),
        ("grok", "x-ai/grok-4", None, None, "x-ai/grok-4"),
    ],
)
def test_effective_model_name_is_canonical_and_idempotent(
    tool: str,
    model: str | None,
    profile: str | None,
    effort: str | None,
    expected: str,
) -> None:
    binding = WorkflowAgentBinding(
        agent_id="A1",
        tool_name=tool,
        model_name=model,
        display_name="Agent 1",
        profile=profile,
        effort=effort,
    )
    resolved = binding.effective_model_name()
    assert resolved == expected
    # Calling again on its own output must never double-encode
    assert (
        resolve_effective_model_name(
            tool,
            resolved,
            profile=profile,
            effort=effort,
        )
        == expected
    )


def test_default_model_effort_guards_on_acp_adapters_and_codex_provider(tmp_path) -> None:
    """Config-option ACP adapters handle default-model+effort without invalid 'default' model RPCs."""
    provider = CodexACPProvider(
        _ProviderConfig(
            tool_name="codex",
            serve_args=["acp", "serve"],
            availability_checker=lambda: True,
            model_style="dynamic",
        )
    )
    assert provider.normalize_model_name("gpt-5.4/high") == "gpt-5.4/high"
    assert provider.normalize_model_name("default/high") == "default/high"
    assert provider.normalize_model_name("default") is None
    assert provider.get_serve_command("gpt-5.4/high") == ("codex", ["acp", "serve"])

    session = object.__new__(SyncACPSession)
    session._acp_session = AsyncMock()
    session._acp_session.set_config_option = AsyncMock(return_value=True)
    session._acp_session.set_model = AsyncMock(return_value=True)

    assert asyncio.run(session._apply_official_codex_selection("default/high")) is True
    session._acp_session.set_config_option.assert_awaited_with("reasoning_effort", "high")
    session._acp_session.set_model.assert_not_awaited()

    session._acp_session.set_config_option.reset_mock()
    assert asyncio.run(session._apply_traex_selection("default/high")) is True
    session._acp_session.set_config_option.assert_awaited_with("reasoning_effort", "high")

    session._acp_session.set_config_option.reset_mock()
    assert asyncio.run(session._apply_dsh_selection('["default","default","high"]')) is True
    session._acp_session.set_config_option.assert_awaited_with(
        "dsh.reasoning_effort", '["effort","high"]'
    )

    session._acp_session.set_config_option.reset_mock()
    assert asyncio.run(session._apply_agy_selection("default/xhigh")) is True
    session._acp_session.set_config_option.assert_awaited_with("thought_level", "xhigh")


def test_run_spec_multi_effort_map_and_end_to_end_passthrough(tmp_path) -> None:
    """Effort/profile/role propagate end-to-end with isolated journal keys and snapshots."""
    b_low = WorkflowAgentBinding(
        agent_id="A1",
        tool_name="codex",
        model_name="gpt-5.4",
        display_name="Worker Low",
        effort="low",
        role="worker",
    )
    b_xhigh = WorkflowAgentBinding(
        agent_id="A2",
        tool_name="codex",
        model_name="gpt-5.4",
        display_name="Reviewer XHigh",
        effort="xhigh",
        role="reviewer",
    )
    b_claude = WorkflowAgentBinding(
        agent_id="A3",
        tool_name="claude",
        model_name="claude-sonnet-4-5",
        display_name="Claude High",
        effort="high",
    )
    reviewer = ReviewAgentBinding(
        provider="workflow",
        tool_name="codex",
        display_name="Reviewer XHigh",
        agent_type="codex",
        model_name="gpt-5.4",
        selection_key="A2",
        use_default_model=False,
    )

    spec = WorkflowRunSpec(
        orchestrator=reviewer,
        reviewers=(reviewer,),
        tool_model_map={},
        task="implement effort passthrough",
        chat_id="chat-m1",
        topic_id=None,
        budget=10,
        deadline=None,
        auto_reviewer=False,
        agent_pool=(b_low, b_xhigh, b_claude),
        orchestrator_agent_id="A1",
    )

    # Multi-config tool `codex` omitted from legacy tool_model_map, unambiguous `claude` preserved
    assert dict(spec.tool_model_map) == {"claude": "claude-sonnet-4-5/high"}
    assert spec.binding_specs["A1"]["effort"] == "low"
    assert spec.binding_specs["A1"]["effective_model"] == "gpt-5.4/low"
    assert spec.binding_specs["A2"]["effort"] == "xhigh"
    assert spec.binding_specs["A2"]["effective_model"] == "gpt-5.4/xhigh"
    with pytest.raises(TypeError):
        spec.binding_specs["A1"]["effort"] = "mutated"  # type: ignore[index]

    reloaded = WorkflowRunSpec.from_dict(spec.to_dict())
    assert dict(reloaded.tool_model_map) == dict(spec.tool_model_map)
    assert reloaded.binding_specs["A2"]["effective_model"] == "gpt-5.4/xhigh"

    # Legacy 5-part key untouched when effort/profile omitted; distinct when effort/profile set
    legacy_key = WorkflowJournal.compute_key("p", "codex", "gpt-5.4", role="worker")
    low_key = WorkflowJournal.compute_key("p", "codex", "gpt-5.4/low", role="worker", effort="low")
    high_key = WorkflowJournal.compute_key("p", "codex", "gpt-5.4/xhigh", role="worker", effort="xhigh")
    assert len({legacy_key, low_key, high_key}) == 3

    engine = WorkflowEngine(chat_id="chat-m1", root_path=str(tmp_path))
    engine._run_spec = spec
    engine._project = WorkflowProject(
        workflow_id="wf-m1",
        status=WorkflowStatus.RUNNING,
        selected_tools=["codex", "claude"],
    )
    engine._state_manager = WorkflowStateManager(engine._project)
    engine._journal = WorkflowJournal(root_path=str(tmp_path), run_id="wf-m1")

    done_payloads: list[dict] = []
    engine._callbacks = WorkflowEngineCallbacks(
        on_agent_done=lambda _label, payload: done_payloads.append(payload)
    )

    session_models: list[str | None] = []

    class FakeSession:
        def send_prompt(self, _prompt, **_kwargs):
            return SimpleNamespace(
                text=f"done:{len(session_models)}",
                output_tokens=10,
                stop_reason="end_turn",
            )

        def close(self):
            return None

    def fake_create_session(**kwargs):
        session_models.append(kwargs.get("model_name"))
        return FakeSession()

    engine._executor = AgentExecutor(cwd=str(tmp_path), cancel_event=threading.Event())
    try:
        with patch("src.agent_session.factory.create_engine_session", side_effect=fake_create_session):
            r_low = engine._handle_agent_call(
                AgentCallParams(prompt="same task", agent_id="A1", label="step-low")
            )
            r_xhigh = engine._handle_agent_call(
                AgentCallParams(prompt="same task", agent_id="A2", role="worker", label="step-xhigh")
            )
            r_low_cached = engine._handle_agent_call(
                AgentCallParams(prompt="same task", agent_id="A1", label="step-low-2")
            )
    finally:
        engine._executor.shutdown(wait=True)

    # Distinct effort never shares cache; identical binding hits cache
    assert r_low.cached is False and r_low.effort == "low" and r_low.role == "worker"
    assert r_low.effective_model == "gpt-5.4/low"
    assert r_xhigh.cached is False and r_xhigh.effort == "xhigh"
    assert r_xhigh.effective_model == "gpt-5.4/xhigh"
    assert r_low_cached.cached is True and r_low_cached.effort == "low"
    assert session_models == ["gpt-5.4/low", "gpt-5.4/xhigh"]

    snapshot = engine._state_manager.snapshot()
    agents = snapshot.phases[0].agents
    assert [(a.agent_id, a.effort, a.effective_model, a.role) for a in agents[:2]] == [
        ("A1", "low", "gpt-5.4/low", "worker"),
        ("A2", "xhigh", "gpt-5.4/xhigh", "worker"),
    ]
    assert all("output" not in payload and "parsed" not in payload for payload in done_payloads)
    assert [(p["agent_id"], p["effort"], p["effective_model"]) for p in done_payloads] == [
        ("A1", "low", "gpt-5.4/low"),
        ("A2", "xhigh", "gpt-5.4/xhigh"),
        ("A1", "low", "gpt-5.4/low"),
    ]


def test_executor_startup_circuit_breaker_isolates_per_binding(tmp_path) -> None:
    """ACPStartupError on one codex effort binding must not blacklist other codex bindings."""
    from src.acp.sync_adapter import ACPStartupError

    executor = AgentExecutor(cwd=str(tmp_path), cancel_event=threading.Event())
    seen_models: list[str | None] = []

    class OkSession:
        def send_prompt(self, _prompt, **_kwargs):
            return SimpleNamespace(text="ok", output_tokens=5, stop_reason="end_turn")

        def close(self):
            return None

    def fake_create_session(**kwargs):
        model = kwargs.get("model_name")
        seen_models.append(model)
        if model == "gpt-5.4/xhigh":
            raise ACPStartupError("bad xhigh config", agent_cmd="codex", agent_args=[], cwd=str(tmp_path))
        return OkSession()

    p_bad = AgentCallParams(
        prompt="bad",
        agent_id="A1",
        tool="codex",
        model="gpt-5.4",
        effort="xhigh",
        effective_model="gpt-5.4/xhigh",
    )
    p_good = AgentCallParams(
        prompt="good",
        agent_id="A2",
        tool="codex",
        model="gpt-5.4",
        effort="low",
        effective_model="gpt-5.4/low",
    )

    try:
        with patch("src.agent_session.factory.create_engine_session", side_effect=fake_create_session):
            res_first_bad = executor.execute(p_bad)
            res_second_bad = executor.execute(p_bad)
            res_good = executor.execute(p_good)
    finally:
        executor.shutdown(wait=True)

    assert res_first_bad.error is not None
    assert "Circuit breaker" in (res_second_bad.error or "")
    assert "ACP" in (res_second_bad.error or "")
    assert res_good.error is None and res_good.output == "ok"
    assert seen_models == ["gpt-5.4/xhigh", "gpt-5.4/low"]


def test_compaction_and_failover_preserve_model_and_effort_for_config_option_acp(tmp_path) -> None:
    """Compaction and failover on codex/traex/dsh/agy preserve _model_name and _explicit_env."""
    assert _extract_model_from_agent_args(["--model", "claude-sonnet-4-5"]) == "claude-sonnet-4-5"
    assert _extract_model_from_agent_args(["--model=gpt-5.4"]) == "gpt-5.4"
    assert _extract_model_from_agent_args(["-c", 'model="Test-O"']) == "Test-O"

    new_args, replaced = _replace_model_in_agent_args(["--model=old-m"], "new-m")
    assert replaced is True and new_args == ["--model=new-m"]

    old = SimpleNamespace(
        _agent_type="traex",
        _cwd=str(tmp_path),
        _agent_cmd="traex",
        _agent_args=["acp", "serve"],
        _model_name="c_o_new_thinking/max/xhigh",
        _explicit_env={"CUSTOM_KEY": "1"},
        _capture_full_tool_content=True,
        start=lambda startup_timeout=20: "sid-old",
        close=lambda: None,
    )
    compaction_kwargs: dict[str, object] = {}

    class Rebuilt:
        def start(self, startup_timeout=20):
            return "sid-new"

        def close(self):
            return None

    def builder(**kwargs):
        compaction_kwargs.update(kwargs)
        return Rebuilt()

    rebuilt = _apply_compaction_once(session=old, session_builder=builder, startup_timeout_s=1.0)
    assert rebuilt is not None
    assert compaction_kwargs["model_name"] == "c_o_new_thinking/max/xhigh"
    assert compaction_kwargs["env"] == {"CUSTOM_KEY": "1"}

    # Failover without CLI model flags on config-option ACP backend (`codex` via npx)
    failover_kwargs: dict[str, object] = {}

    class FailoverBase:
        def __init__(self):
            self._agent_type = "codex"
            self._cwd = str(tmp_path)
            self._agent_cmd = "npx"
            self._agent_args = ["--yes", "@agentclientprotocol/codex-acp@1.2.0"]
            self._model_name = "gpt-5.4/high"
            self._explicit_env = {"OPENAI_BASE": "https://example.invalid"}
            self._capture_full_tool_content = True
            self.calls = 0

        def close(self):
            return None

        def send_prompt(self, _text, **_kwargs):
            self.calls += 1
            raise RuntimeError("loop detected")

    class FailoverReplacement:
        def __init__(self, **kwargs):
            failover_kwargs.update(kwargs)

        def start(self, startup_timeout=20):
            return "sid-failover"

        def close(self):
            return None

        def send_prompt(self, _text, **_kwargs):
            return SimpleNamespace(stop_reason="end_turn", text="failover-ok")

    settings = SimpleNamespace(
        model_failure_compaction_enabled=True,
        model_failure_compaction_loop_window_s=180.0,
        model_failure_compaction_loop_max=2,
        model_failure_failover_map="gpt-5.4/high:gpt-5.3-codex/medium",
        acp_startup_timeout=5,
    )
    with (
        patch("src.agent_session.wrappers.get_settings", return_value=settings),
        patch("src.agent_session.wrappers.SyncACPSession", FailoverReplacement),
    ):
        wrapper = ModelFailureAwareSession(FailoverBase())
        res = wrapper.send_prompt("hello")

    assert res.text == "failover-ok"
    assert failover_kwargs["model_name"] == "gpt-5.3-codex/medium"
    assert failover_kwargs["env"] == {"OPENAI_BASE": "https://example.invalid"}

    # Legacy session_builder without model_name/env still works via TypeError fallback
    def legacy_builder(agent_type, cwd, agent_cmd, agent_args, capture_full_tool_content):
        assert agent_type == "traex"
        assert cwd == str(tmp_path)
        assert agent_cmd == "traex"
        assert capture_full_tool_content is True
        return Rebuilt()

    assert _apply_compaction_once(session=old, session_builder=legacy_builder, startup_timeout_s=1.0) is not None


def test_validate_traex_agy_dsh_profiles_and_direct_engine_resolution(tmp_path) -> None:
    """Validate Traex/AGY/DSH profiles & efforts and direct engine call resolution without run_spec."""
    caps = {
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("c_o_new_thinking",),
            efforts_by_model={"c_o_new_thinking": ("low", "high", "xhigh")},
            profiles_by_model={"c_o_new_thinking": ("standard", "max")},
        ),
        "agy": ToolCapability(
            tool_name="agy",
            display_name="AGY",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gemini-4-neon", "gemini-pro-orchestrator-neon"),
            efforts_by_model={
                "gemini-4-neon": ("low", "high"),
                "gemini-pro-orchestrator-neon": ("low", "high"),
            },
            profiles_by_model={"gemini-pro-orchestrator-neon": ("low", "high")},
        ),
        "dsh": ToolCapability(
            tool_name="dsh",
            display_name="DSH",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("deepseek/deepseek-reasoner",),
            efforts_by_model={"deepseek/deepseek-reasoner": ("default", "off", "high")},
        ),
    }

    with patch("src.acp.traex_selection.load_traex_model_metadata", return_value=()):
        assert validate_agent_binding_spec("traex", model_name="c_o_new_thinking", profile="turbo", capabilities=caps)[0] is False
        assert validate_agent_binding_spec("traex", model_name="c_o_new_thinking", profile="max", effort="xhigh", capabilities=caps)[0] is True

    assert validate_agent_binding_spec("agy", model_name="gemini-4-neon", profile="high", capabilities=caps)[0] is False
    assert validate_agent_binding_spec("agy", model_name="gemini-4-neon/invalid", capabilities=caps)[0] is False
    assert validate_agent_binding_spec("agy", model_name="gemini-pro-orchestrator-neon", profile="high", effort="low", capabilities=caps)[0] is True

    assert validate_agent_binding_spec("dsh", model_name="badformat", capabilities=caps)[0] is False
    assert validate_agent_binding_spec("dsh", model_name="deepseek/deepseek-reasoner", effort="off", capabilities=caps)[0] is True

    # Direct engine._handle_agent_call without run_spec still resolves effective_model
    engine = WorkflowEngine(chat_id="chat-direct", root_path=str(tmp_path))
    engine._project = WorkflowProject(workflow_id="wf-direct", status=WorkflowStatus.RUNNING)
    engine._state_manager = WorkflowStateManager(engine._project)
    engine._executor = MagicMock()
    engine._executor.execute.side_effect = lambda params, **_kw: AgentCallResult(
        output="ok",
        stop_reason="end_turn",
        agent_id=params.agent_id,
        tool=params.tool,
        model=params.model,
        effort=params.effort,
        profile=params.profile,
        effective_model=params.effective_model,
    )
    res = engine._handle_agent_call(
        AgentCallParams(prompt="direct", tool="codex", model="gpt-5.4", effort="xhigh")
    )
    assert res.effective_model == "gpt-5.4/xhigh"
    assert res.effort == "xhigh"


def test_run_spec_rejects_spoofed_reviewer_and_preserves_effective_model_on_cold_reload(
    tmp_path,
) -> None:
    """Reviewer selection_key cannot bypass tool/model checks; from_dict preserves resolved effective_model."""
    pool = (
        WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex Low",
            effort="low",
        ),
        WorkflowAgentBinding(
            agent_id="A2",
            tool_name="codex",
            model_name="gpt-5.4",
            display_name="Codex XHigh",
            effort="xhigh",
        ),
    )
    valid_reviewer = ReviewAgentBinding(
        provider="workflow",
        tool_name="codex",
        display_name="Codex Low",
        agent_type="codex",
        model_name="gpt-5.4/low",
        selection_key="A1",
        use_default_model=False,
    )
    spoofed_tool_reviewer = ReviewAgentBinding(
        provider="workflow",
        tool_name="claude",
        display_name="Spoofed Claude",
        agent_type="claude",
        model_name="claude-opus",
        selection_key="A1",
        use_default_model=False,
    )
    mismatched_effort_reviewer = ReviewAgentBinding(
        provider="workflow",
        tool_name="codex",
        display_name="Mismatched Effort",
        agent_type="codex",
        model_name="gpt-5.4/xhigh",
        selection_key="A1",
        use_default_model=False,
    )

    for bad_reviewer in (spoofed_tool_reviewer, mismatched_effort_reviewer):
        with pytest.raises(ValueError, match="outside agent_pool"):
            WorkflowRunSpec(
                orchestrator=valid_reviewer,
                reviewers=(bad_reviewer,),
                tool_model_map={},
                task="reject spoofed reviewer",
                chat_id="chat-spoof",
                topic_id=None,
                budget=4,
                deadline=None,
                auto_reviewer=False,
                agent_pool=pool,
                orchestrator_agent_id="A1",
            )

    # Warm ACP cache -> model_name=None, effort="high" freezes gpt-5.4/high
    codex_options = [
        ACPModelOption(
            name="gpt-5.4",
            is_default=True,
            reasoning_efforts=("low", "medium", "high", "xhigh"),
        )
    ]
    with _cache_lock:
        _probe_cache[("codex", "")] = (time.time(), codex_options)
    try:
        default_binding = WorkflowAgentBinding(
            agent_id="A1",
            tool_name="codex",
            model_name=None,
            display_name="Codex Default High",
            effort="high",
        )
        warm_reviewer = ReviewAgentBinding(
            provider="workflow",
            tool_name="codex",
            display_name="Codex Default High",
            agent_type="codex",
            model_name="gpt-5.4/high",
            selection_key="A1",
            use_default_model=False,
        )
        spec = WorkflowRunSpec(
            orchestrator=warm_reviewer,
            reviewers=(warm_reviewer,),
            tool_model_map={},
            task="preserve warm effective model",
            chat_id="chat-warm",
            topic_id=None,
            budget=4,
            deadline=None,
            auto_reviewer=False,
            agent_pool=(default_binding,),
            orchestrator_agent_id="A1",
        )
        assert spec.binding_specs["A1"]["effective_model"] == "gpt-5.4/high"
        serialized = spec.to_dict()
    finally:
        with _cache_lock:
            _probe_cache.pop(("codex", ""), None)

    # Cold ACP cache reload still preserves gpt-5.4/high and accepts warm_reviewer
    reloaded = WorkflowRunSpec.from_dict(serialized)
    assert reloaded.binding_specs["A1"]["effective_model"] == "gpt-5.4/high"
    assert dict(reloaded.tool_model_map) == {"codex": "gpt-5.4/high"}

    # Tampered binding_specs cannot change effort or explicit model
    tampered = dict(serialized)
    tampered["auto_reviewer"] = True
    tampered["reviewers"] = []
    tampered["binding_specs"] = {
        "A1": {**serialized["binding_specs"]["A1"], "effective_model": "gpt-5.4/low"}
    }
    tampered_reloaded = WorkflowRunSpec.from_dict(tampered)
    assert tampered_reloaded.binding_specs["A1"]["effective_model"] == "default/high"


def test_validate_agent_binding_spec_accepts_default_sentinel_with_warm_models() -> None:
    """Warm cap.models accepts 'default' and canonical default/effort strings while validating effort/profile."""
    caps = {
        "codex": ToolCapability(
            tool_name="codex",
            display_name="Codex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gpt-5.4",),
            efforts_by_model={"gpt-5.4": ("low", "medium", "high", "xhigh")},
        ),
        "traex": ToolCapability(
            tool_name="traex",
            display_name="Traex",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("c_o_new_thinking",),
            efforts_by_model={"c_o_new_thinking": ("low", "high", "xhigh")},
            profiles_by_model={"c_o_new_thinking": ("standard", "max")},
        ),
        "agy": ToolCapability(
            tool_name="agy",
            display_name="AGY",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("gemini-4-neon",),
            efforts_by_model={"gemini-4-neon": ("low", "high")},
        ),
        "dsh": ToolCapability(
            tool_name="dsh",
            display_name="DSH",
            available=True,
            supports_model=True,
            supports_effort=True,
            models=("deepseek/deepseek-reasoner",),
            efforts_by_model={"deepseek/deepseek-reasoner": ("default", "off", "high")},
        ),
        "coco": ToolCapability(
            tool_name="coco",
            display_name="Coco",
            available=True,
            supports_model=True,
            supports_effort=False,
            models=("openai/gpt-5", "anthropic/claude-sonnet"),
        ),
    }

    with patch("src.acp.traex_selection.load_traex_model_metadata", return_value=()):
        assert validate_agent_binding_spec("codex", model_name="default", effort="high", capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("codex", model_name="default/high", capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("traex", model_name="default/standard/high", capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("agy", model_name="default/high", capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("dsh", model_name="default/default", effort="high", capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("dsh", model_name='["default","default","high"]', capabilities=caps) == (True, None)
        assert validate_agent_binding_spec("dsh", model_name="default", effort="high", capabilities=caps) == (True, None)

        # Invalid effort/profile still rejected even with default base model
        assert validate_agent_binding_spec("codex", model_name="default/bogus", capabilities=caps)[0] is False
        assert validate_agent_binding_spec("codex", model_name="default", effort="bogus", capabilities=caps)[0] is False
        assert validate_agent_binding_spec("traex", model_name="default/turbo/high", capabilities=caps)[0] is False
        assert validate_agent_binding_spec("agy", model_name="default", effort="bogus", capabilities=caps)[0] is False
        assert validate_agent_binding_spec("dsh", model_name='["default","default","bogus"]', capabilities=caps)[0] is False

    # OpenRouter slash IDs remain intact on model-only backends
    assert validate_agent_binding_spec("coco", model_name="anthropic/claude-sonnet", capabilities=caps) == (True, None)
    assert validate_agent_binding_spec("coco", model_name="openai/gpt-5", capabilities=caps) == (True, None)


def test_codex_failover_strips_cli_effort_and_native_codex_applies_acp_config(tmp_path) -> None:
    """Codex failover strips /effort & default from CLI argv; native codex acp serve applies ACP config options."""
    captured: list[dict[str, object]] = []

    class NativeCodexBase:
        def __init__(self, args: list[str], model: str):
            self._agent_type = "codex"
            self._cwd = str(tmp_path)
            self._agent_cmd = "codex"
            self._agent_args = list(args)
            self._model_name = model
            self._explicit_env = None
            self._capture_full_tool_content = False

        def close(self):
            return None

        def send_prompt(self, _text, **_kwargs):
            raise RuntimeError("loop detected")

    class Replacement:
        def __init__(self, **kwargs):
            captured.append(dict(kwargs))

        def start(self, startup_timeout=20):
            return "sid-next"

        def close(self):
            return None

        def send_prompt(self, _text, **_kwargs):
            return SimpleNamespace(stop_reason="end_turn", text="ok")

    settings = SimpleNamespace(
        model_failure_compaction_enabled=True,
        model_failure_compaction_loop_window_s=180.0,
        model_failure_compaction_loop_max=2,
        model_failure_failover_map="gpt-5.4/high:gpt-5.3-codex/medium,gpt-5.3-codex/medium:default/low",
        acp_startup_timeout=5,
    )
    with (
        patch("src.agent_session.wrappers.get_settings", return_value=settings),
        patch("src.agent_session.wrappers.SyncACPSession", Replacement),
    ):
        res1 = ModelFailureAwareSession(
            NativeCodexBase(["acp", "serve", "-c", "model.name=gpt-5.4"], "gpt-5.4/high")
        ).send_prompt("hi")
        res2 = ModelFailureAwareSession(
            NativeCodexBase(["acp", "serve", "-c", "model.name=gpt-5.3-codex"], "gpt-5.3-codex/medium")
        ).send_prompt("hi")

    assert res1.text == "ok" and res2.text == "ok"
    assert captured[0]["agent_args"] == ["acp", "serve", "-c", "model.name=gpt-5.3-codex"]
    assert captured[0]["model_name"] == "gpt-5.3-codex/medium"
    assert captured[1]["agent_args"] == ["acp", "serve"]
    assert captured[1]["model_name"] == "default/low"

    # Native `codex acp serve` (without @agentclientprotocol/codex-acp) applies model & effort over ACP
    native_sync = SyncACPSession(
        agent_type="codex",
        cwd=str(tmp_path),
        agent_cmd="codex",
        agent_args=["acp", "serve"],
        model_name="gpt-5.4/xhigh",
    )
    assert native_sync._uses_official_codex_acp() is False
    assert native_sync._uses_codex_acp() is True

    async_session = AsyncMock()
    async_session.start = AsyncMock(return_value="sid-native")
    async_session.set_config_option = AsyncMock(return_value=True)
    async_session.set_model = AsyncMock(return_value=True)
    async_session.close = AsyncMock()
    with patch("src.acp.sync_adapter.ACPSession", return_value=async_session):
        assert asyncio.run(native_sync._start_session()) == "sid-native"

    assert async_session.set_config_option.await_args_list == [
        (("model", "gpt-5.4"), {}),
        (("reasoning_effort", "xhigh"), {}),
    ]

    # Native provider with config_c help blob forwards model_reasoning_effort too
    provider_c = CodexACPProvider(
        _ProviderConfig(
            tool_name="codex",
            serve_args=["acp", "serve"],
            availability_checker=lambda: True,
            model_style="dynamic",
            help_blob_loader=lambda: "Usage: codex acp serve -c model.name=<id>",
        )
    )
    assert provider_c.get_serve_command("gpt-5.4/xhigh") == (
        "codex",
        ["acp", "serve", "-c", "model.name=gpt-5.4", "-c", 'model_reasoning_effort="xhigh"'],
    )
    assert provider_c.get_serve_command("default/xhigh") == (
        "codex",
        ["acp", "serve", "-c", 'model_reasoning_effort="xhigh"'],
    )
