import asyncio
import json
import time
from types import SimpleNamespace

import src.acp.helper as _helper_mod
from src.acp.helper import fetch_acp_models
from src.coco_model.manager import DEFAULT_MODELS
from src.ttadk.models import ACPModelOption


def test_fetch_codex_models_timeout_returns_empty_without_adapter_models(monkeypatch, tmp_path):
    async def slow_probe(_tool_name, _cwd, _current_model):
        await asyncio.sleep(1)
        return []

    (tmp_path / ".codex").mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("src.acp.helper.probe_acp_models", slow_probe)

    started = time.monotonic()
    models = fetch_acp_models(
        "codex",
        cwd="/tmp/ghostap",
        current_model="current-fast-fallback",
        probe_timeout=0.1,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert models == []


def test_fetch_codex_models_does_not_use_local_codex_cache_after_empty_probe(monkeypatch, tmp_path):
    probe_calls = []

    async def empty_probe(*args, **_kwargs):
        probe_calls.append(args)
        return []

    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "models_cache.json").write_text(
        """
        {
          "models": [
            {"slug": "gpt-5.5", "display_name": "GPT-5.5", "visibility": "list", "priority": 0},
            {"slug": "gpt-5.4", "display_name": "GPT-5.4", "visibility": "list", "priority": 1}
          ]
        }
        """,
        encoding="utf-8",
    )

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("src.acp.helper.probe_acp_models", empty_probe)

    models = fetch_acp_models("codex", cwd="/tmp/ghostap", probe_timeout=0.1)

    assert len(probe_calls) == 1
    assert models == []


def test_fetch_codex_models_does_not_use_local_codex_cache_for_current_model_after_empty_probe(monkeypatch, tmp_path):
    probe_calls = []

    async def empty_probe(*args, **_kwargs):
        probe_calls.append(args)
        return []

    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5.5"\n', encoding="utf-8")
    (codex_home / "models_cache.json").write_text(
        """
        {
          "models": [
            {"slug": "gpt-5.5", "display_name": "GPT-5.5", "visibility": "list", "priority": 0},
            {"slug": "gpt-5.4", "display_name": "GPT-5.4", "visibility": "list", "priority": 1},
            {"slug": "gpt-5.3", "display_name": "GPT-5.3", "visibility": "hidden", "priority": 2}
          ]
        }
        """,
        encoding="utf-8",
    )

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("src.acp.helper.probe_acp_models", empty_probe)

    models = fetch_acp_models(
        "codex",
        cwd="/tmp/ghostap",
        current_model="gpt-5.4",
        probe_timeout=0.1,
    )

    assert len(probe_calls) == 1
    assert models == []


def test_fetch_coco_models_timeout_uses_static_defaults(monkeypatch):
    """When probe times out AND CocoModelManager only has static defaults
    cached, fetch_acp_models must degrade to DEFAULT_MODELS (no infinite
    re-probe loop). The dedicated cached-model-bypass test below covers the
    happy path where manager has real ACP models cached."""

    async def slow_probe(_tool_name, _cwd, _current_model):
        await asyncio.sleep(1)
        return []

    from src.coco_model.models import CocoModel, ModelListResult

    class FakeCocoManager:
        def get_current_model(self):
            return "gpt-4.1"

        def get_models(self):
            # Return the same static defaults — fetch_acp_models should treat
            # this as "probe needed" and fall through, not loop.
            return ModelListResult(
                models=[CocoModel(name=m.name, description=m.description) for m in DEFAULT_MODELS],
                cached=False,
            )

    monkeypatch.setattr("src.acp.helper.probe_acp_models", slow_probe)
    monkeypatch.setattr("src.coco_model.get_coco_model_manager", lambda: FakeCocoManager())

    started = time.monotonic()
    models = fetch_acp_models("coco", cwd="/tmp/ghostap", probe_timeout=0.1)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert [m.name for m in models] == [m.name for m in DEFAULT_MODELS]
    assert [m.name for m in models if m.is_default] == ["gpt-4.1"]


def test_fetch_coco_models_uses_manager_cache_when_real_models_present(monkeypatch):
    """When CocoModelManager already cached real ACP models, fetch_acp_models
    must return those (and skip the fresh probe) so selectors share the same
    model list source."""

    real_models = ["GPT-5.2", "GPT-5.4", "Gemini-3.1-Pro-Preview", "Test-O-New"]

    async def probe_should_not_run(*_a, **_k):  # pragma: no cover
        raise AssertionError("probe should be skipped when manager has real models")

    from src.coco_model.models import CocoModel, ModelListResult

    class FakeCocoManager:
        def get_current_model(self):
            return "GPT-5.4"

        def get_models(self):
            return ModelListResult(
                models=[CocoModel(name=name, description=name) for name in real_models],
                cached=True,
            )

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe_should_not_run)
    monkeypatch.setattr("src.coco_model.get_coco_model_manager", lambda: FakeCocoManager())

    started = time.monotonic()
    models = fetch_acp_models("coco", cwd="/tmp/ghostap")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert [m.name for m in models] == real_models
    assert [m.name for m in models if m.is_default] == ["GPT-5.4"]


# ---------------------------------------------------------------------------
# Non-coco ACP probe cache tests
# ---------------------------------------------------------------------------


def test_codex_success_cache_remains_fresh_for_thirty_minutes(monkeypatch):
    key = _helper_mod._probe_key("codex", "/repo")
    monkeypatch.setitem(
        _helper_mod._acp_probe_cache,
        key,
        (
            _helper_mod._time.time() - 600,
            [ACPModelOption(name="gpt-test", description="test", is_default=True)],
        ),
    )

    async def probe_must_not_run(*_args, **_kwargs):
        raise AssertionError("ten-minute-old Codex cache should still be fresh")

    monkeypatch.setattr(_helper_mod, "probe_acp_models", probe_must_not_run)
    assert [m.name for m in fetch_acp_models("codex", cwd="/repo")] == ["gpt-test"]


def test_background_preheat_deduplicates_codex(monkeypatch):
    calls = []

    def fake_fetch(tool_name, *, cwd):
        calls.append((tool_name, cwd))
        return [
            ACPModelOption(
                name="gpt-test",
                description="test",
                is_default=True,
            )
        ]

    monkeypatch.setattr(_helper_mod, "fetch_acp_models", fake_fetch)

    thread = _helper_mod.kickoff_acp_model_preheat(
        ["codex", "codex"],
        "/repo",
    )
    assert thread is not None
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert calls == [("codex", "/repo")]


def test_fetch_acp_models_non_coco_caches_successful_probe(monkeypatch):
    """Successful probe for non-coco tools is cached in _acp_probe_cache."""
    from src.ttadk.models import ACPModelOption

    probe_results = [
        ACPModelOption(name="model-a", description="A", is_default=True),
        ACPModelOption(name="model-b", description="B", is_default=False),
    ]

    async def fake_probe(_tool_name, _cwd, _current_model):
        return probe_results

    monkeypatch.setattr("src.acp.helper.probe_acp_models", fake_probe)
    # Clear cache before test
    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()

    models = fetch_acp_models("aiden", cwd="/tmp/ghostap", current_model="model-a")

    assert [m.name for m in models] == ["model-a", "model-b"]
    # Verify cache was populated (keyed by (tool, cwd))
    key = _helper_mod._probe_key("aiden", "/tmp/ghostap")
    assert key in _helper_mod._acp_probe_cache
    _ts, cached_models = _helper_mod._acp_probe_cache[key]
    assert [m.name for m in cached_models] == ["model-a", "model-b"]


def test_fetch_acp_models_non_coco_uses_cache_on_probe_failure(monkeypatch):
    """When probe fails for non-coco tool, cached result is used as fallback."""
    from src.ttadk.models import ACPModelOption

    # Pre-populate cache
    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_cache[_helper_mod._probe_key("codex", "/tmp/ghostap")] = (
        _helper_mod._time.time(),
        [
            ACPModelOption(name="cached-1", description="C1", is_default=True),
            ACPModelOption(name="cached-2", description="C2", is_default=False),
        ],
    )

    async def failing_probe(_tool_name, _cwd, _current_model):
        raise RuntimeError("network error")

    monkeypatch.setattr("src.acp.helper.probe_acp_models", failing_probe)

    models = fetch_acp_models("codex", cwd="/tmp/ghostap", current_model="cached-2")

    assert [m.name for m in models] == ["cached-1", "cached-2"]
    # current_model should be re-marked as default
    assert [m.name for m in models if m.is_default] == ["cached-2"]


def test_fetch_acp_models_non_coco_expired_cache_not_used(monkeypatch):
    """Expired cache entries are not returned; falls back to current_model."""
    from src.ttadk.models import ACPModelOption

    # Pre-populate cache with expired entry (TTL + 10s ago)
    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    expired_ts = _helper_mod._time.time() - _helper_mod._ACP_PROBE_CACHE_TTL - 10
    _helper_mod._acp_probe_cache[_helper_mod._probe_key("gemini", "/tmp/ghostap")] = (
        expired_ts,
        [ACPModelOption(name="old-model", description="Old", is_default=True)],
    )

    async def failing_probe(_tool_name, _cwd, _current_model):
        raise RuntimeError("timeout")

    monkeypatch.setattr("src.acp.helper.probe_acp_models", failing_probe)

    models = fetch_acp_models("gemini", cwd="/tmp/ghostap", current_model="fallback-model")

    # Should not return expired cache; should fall back to current_model only
    assert [m.name for m in models] == ["fallback-model"]
    assert models[0].is_default is True


# ---------------------------------------------------------------------------
# Single-flight coalescing + negative cache tests
# (the fix for "model selection is slow": concurrent probes were not
#  de-duplicated, and a tool that times out every time — e.g. claude — was
#  re-probed on every card click)
# ---------------------------------------------------------------------------


def test_fetch_acp_models_concurrent_probes_coalesce_to_single_call(monkeypatch):
    """Concurrent lookups for the same (tool, cwd) run the probe only once."""
    import threading

    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    call_count = 0
    release = threading.Event()

    async def slow_probe(_tool_name, _cwd, _current_model):
        nonlocal call_count
        call_count += 1
        # Hold the leader here so the other threads pile up as waiters.
        release.wait(timeout=5)
        return [ACPModelOption(name="m-1", description="M1", is_default=True)]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", slow_probe)

    results: list[list] = []
    results_lock = threading.Lock()

    def worker():
        models = fetch_acp_models("aiden", cwd="/tmp/ghostap", probe_timeout=5.0)
        with results_lock:
            results.append(models)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    # Give the leader a moment to enter the probe, then let it finish.
    time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(timeout=10)

    assert call_count == 1, f"probe should run once, ran {call_count} times"
    assert len(results) == 5
    for r in results:
        assert [m.name for m in r] == ["m-1"]


def test_fetch_acp_models_negative_cache_skips_reprobe(monkeypatch):
    """After an empty/timed-out probe, the tool is not re-probed within TTL."""
    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    call_count = 0

    async def empty_probe(_tool_name, _cwd, _current_model):
        nonlocal call_count
        call_count += 1
        return []

    monkeypatch.setattr("src.acp.helper.probe_acp_models", empty_probe)

    first = fetch_acp_models("claude", cwd="/tmp/ghostap", probe_timeout=1.0)
    second = fetch_acp_models("claude", cwd="/tmp/ghostap", probe_timeout=1.0)
    # A current_model on a neg-cache hit degrades to that model (no live probe).
    third = fetch_acp_models("claude", cwd="/tmp/ghostap", current_model="x", probe_timeout=1.0)

    assert first == []
    assert second == []
    assert [m.name for m in third] == ["x"]
    # Only the first call should have hit the live probe; the rest are served
    # by the negative cache.
    assert call_count == 1, f"expected 1 live probe, got {call_count}"


def test_fetch_acp_models_negative_cache_expires_and_reprobes(monkeypatch):
    """Once the negative-cache TTL passes, the tool is probed again."""
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    call_count = 0

    async def probe(_tool_name, _cwd, _current_model):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return []
        return [ACPModelOption(name="recovered", description="R", is_default=True)]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe)

    assert fetch_acp_models("gemini", cwd="/tmp/ghostap", probe_timeout=1.0) == []
    # Force the negative-cache entry to look expired.
    key = _helper_mod._probe_key("gemini", "/tmp/ghostap")
    _helper_mod._acp_neg_cache[key] = (
        _helper_mod._time.time() - _helper_mod._ACP_NEG_CACHE_TTL - 1
    )

    models = fetch_acp_models("gemini", cwd="/tmp/ghostap", probe_timeout=1.0)
    assert [m.name for m in models] == ["recovered"]
    assert call_count == 2


def test_fetch_acp_models_cache_hit_returns_independent_copies(monkeypatch):
    """A cache hit must not let one caller's is_default leak into the cache
    or into another caller's list."""
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    async def probe(_tool_name, _cwd, _current_model):
        return [
            ACPModelOption(name="alpha", description="A", is_default=True),
            ACPModelOption(name="beta", description="B", is_default=False),
        ]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe)

    # Prime the cache.
    fetch_acp_models("aiden", cwd="/tmp/ghostap", probe_timeout=1.0)

    a = fetch_acp_models("aiden", cwd="/tmp/ghostap", current_model="beta")
    b = fetch_acp_models("aiden", cwd="/tmp/ghostap", current_model="alpha")

    assert [m.name for m in a if m.is_default] == ["beta"]
    assert [m.name for m in b if m.is_default] == ["alpha"]
    # The two result lists are independent objects.
    assert a[0] is not b[0]
    # The shared cache entry is untouched by per-caller default marking.
    key = _helper_mod._probe_key("aiden", "/tmp/ghostap")
    _ts, cached = _helper_mod._acp_probe_cache[key]
    assert [m.name for m in cached if m.is_default] == ["alpha"]


def test_first_probe_current_model_does_not_poison_shared_default_cache(monkeypatch):
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    seen_probe_current_models: list[str | None] = []

    async def probe(_tool_name, _cwd, current_model):
        seen_probe_current_models.append(current_model)
        return [
            ACPModelOption(name="provider-default", is_default=True),
            ACPModelOption(name="project-current", is_default=False),
        ]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe)

    project_models = fetch_acp_models(
        "aiden",
        cwd="/tmp/ghostap",
        current_model="project-current",
        probe_timeout=1.0,
    )
    default_models = fetch_acp_models("aiden", cwd="/tmp/ghostap")

    assert seen_probe_current_models == [None]
    assert [model.name for model in project_models if model.is_default] == [
        "project-current"
    ]
    assert [model.name for model in default_models if model.is_default] == [
        "provider-default"
    ]


def test_fetch_codex_models_marks_composite_current_model_by_base(monkeypatch):
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    async def probe(_tool_name, _cwd, _current_model):
        return [
            ACPModelOption(
                name="gpt-5.6-sol",
                description="GPT-5.6-Sol",
                is_default=True,
                reasoning_efforts=("low", "high", "max"),
                adapted_reasoning_effort="high",
            ),
            ACPModelOption(
                name="gpt-5.5",
                description="GPT-5.5",
                is_default=False,
                reasoning_efforts=("low", "high"),
                adapted_reasoning_effort="high",
            ),
        ]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe)

    fetch_acp_models("codex", cwd="/tmp/ghostap", probe_timeout=1.0)
    models = fetch_acp_models(
        "codex",
        cwd="/tmp/ghostap",
        current_model="gpt-5.6-sol/max",
    )

    assert [model.name for model in models if model.is_default] == [
        "gpt-5.6-sol"
    ]


def test_unknown_current_model_keeps_provider_default(monkeypatch):
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    async def probe(_tool_name, _cwd, _current_model):
        return [
            ACPModelOption(name="provider-default", is_default=True),
            ACPModelOption(name="other", is_default=False),
        ]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe)

    models = fetch_acp_models(
        "aiden",
        cwd="/tmp/ghostap",
        current_model="removed-model",
        probe_timeout=1.0,
    )

    assert [model.name for model in models if model.is_default] == [
        "provider-default"
    ]


def test_removed_effort_keeps_existing_base_model_selected(monkeypatch):
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    async def probe(_tool_name, _cwd, _current_model):
        return [
            ACPModelOption(
                name="gpt-5.5",
                is_default=True,
                reasoning_efforts=("low", "high"),
                adapted_reasoning_effort="high",
            ),
            ACPModelOption(
                name="gpt-5.6-sol",
                is_default=False,
                reasoning_efforts=("low", "high", "max"),
                adapted_reasoning_effort="high",
            ),
        ]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe)

    models = fetch_acp_models(
        "codex",
        cwd="/tmp/ghostap",
        current_model="gpt-5.6-sol/ultra",
        probe_timeout=1.0,
    )

    assert [model.name for model in models if model.is_default] == [
        "gpt-5.6-sol"
    ]


def test_fetch_acp_models_different_cwd_probed_separately(monkeypatch):
    """Cache/single-flight are keyed by (tool, cwd), so distinct cwds probe
    independently."""
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()
    _helper_mod._acp_neg_cache.clear()
    _helper_mod._acp_probe_inflight.clear()

    seen_cwds: list[str] = []

    async def probe(_tool_name, cwd, _current_model):
        seen_cwds.append(cwd)
        return [ACPModelOption(name=f"m-{cwd}", description="M", is_default=True)]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", probe)

    fetch_acp_models("aiden", cwd="/repo/a", probe_timeout=1.0)
    fetch_acp_models("aiden", cwd="/repo/b", probe_timeout=1.0)
    # Repeat first cwd — should be served from cache, not re-probed.
    fetch_acp_models("aiden", cwd="/repo/a", probe_timeout=1.0)

    assert seen_cwds == ["/repo/a", "/repo/b"]


def test_fetch_codex_models_prefers_adapter_probe_over_local_codex_cache(monkeypatch, tmp_path):
    """Codex model picker must reflect the running ACP adapter, not stale CLI cache."""
    from src.acp import helper as helper_mod
    from src.acp.helper import fetch_acp_models
    from src.ttadk.models import ACPModelOption

    helper_mod._acp_probe_cache.clear()
    helper_mod._acp_neg_cache.clear()
    helper_mod._acp_probe_inflight.clear()

    cache_dir = tmp_path / ".codex"
    cache_dir.mkdir()
    (cache_dir / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "stale-local-only",
                        "display_name": "Stale local model",
                        "visibility": "list",
                        "priority": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    async def fake_probe(tool_name, cwd, current_model):
        assert tool_name == "codex"
        assert cwd == "/repo"
        assert current_model is None
        return [
            ACPModelOption(
                name="gpt-5.6-sol",
                description="GPT-5.6 Sol from adapter",
                is_default=True,
            )
        ]

    monkeypatch.setattr("src.acp.helper.Path.home", lambda: tmp_path)
    monkeypatch.setattr("src.acp.helper.probe_acp_models", fake_probe)

    models = fetch_acp_models("codex", cwd="/repo", current_model="gpt-5.6-sol", probe_timeout=1.0)

    assert [m.name for m in models] == ["gpt-5.6-sol"]
    assert models[0].is_default is True


# ---------------------------------------------------------------------------
# config_options model extraction tests (traex-style providers)
# ---------------------------------------------------------------------------


def test_extract_models_from_config_options_basic():
    """Models are extracted from config_options when available_models is empty."""
    from src.acp.helper import _extract_models_from_config_options

    class MockOption:
        def __init__(self, name, value, description=None):
            self.name = name
            self.value = value
            self.description = description
            self.field_meta = None

    class MockRoot:
        def __init__(self, category, current_value, options):
            self.category = category
            self.current_value = current_value
            self.options = options
            self.field_meta = None
            self.id = "model"
            self.name = "Model"
            self.type = "select"

    class MockConfigOption:
        def __init__(self, root):
            self.root = root

    class MockResp:
        def __init__(self, config_options):
            self.config_options = config_options

    options = [
        MockOption("Test-O-New-Thinking", "c_o_new_thinking", "200K context"),
        MockOption("GPT-5.5", "gpt-5.5", "272K context"),
        MockOption("Doubao-Seed-Code", "Doubao_1_6", "116K context"),
    ]
    root = MockRoot("model", "c_o_new_thinking", options)
    resp = MockResp([MockConfigOption(root)])

    result = _extract_models_from_config_options(resp, "c_o_new_thinking")

    assert len(result) == 3
    assert result[0].name == "c_o_new_thinking"
    assert result[0].description == "Test-O-New-Thinking"
    assert result[0].is_default is True
    assert result[1].name == "gpt-5.5"
    assert result[1].is_default is False


def test_extract_models_from_config_options_skips_non_model_category():
    """Only config_options with category='model' are used."""
    from src.acp.helper import _extract_models_from_config_options

    class MockOption:
        def __init__(self, name, value):
            self.name = name
            self.value = value
            self.description = None
            self.field_meta = None

    class MockRoot:
        def __init__(self, category, current_value, options):
            self.category = category
            self.current_value = current_value
            self.options = options

    class MockConfigOption:
        def __init__(self, root):
            self.root = root

    class MockResp:
        def __init__(self, config_options):
            self.config_options = config_options

    mode_root = MockRoot("mode", "default", [MockOption("Default", "default")])
    resp = MockResp([MockConfigOption(mode_root)])

    result = _extract_models_from_config_options(resp, "")
    assert result == []


def test_probe_acp_models_falls_back_to_config_options(monkeypatch):
    """probe_acp_models returns models from config_options when
    available_models is empty (traex behavior)."""
    from src.ttadk.models import ACPModelOption

    _helper_mod._acp_probe_cache.clear()

    async def fake_probe(_tool_name, _cwd, _current_model):
        class FakeOption:
            def __init__(self, name, value, description=None):
                self.name = name
                self.value = value
                self.description = description
                self.field_meta = None

        class FakeRoot:
            category = "model"
            current_value = "c_o_new_thinking"
            options = [
                FakeOption("Test-O-New", "c_o_new_thinking", "200K context"),
                FakeOption("GPT-5.5", "gpt-5.5", "272K context"),
            ]

        class FakeConfigOption:
            root = FakeRoot()

        return [
            ACPModelOption(name="c_o_new_thinking", description="Test-O-New", is_default=True),
            ACPModelOption(name="gpt-5.5", description="GPT-5.5", is_default=False),
        ]

    monkeypatch.setattr("src.acp.helper.probe_acp_models", fake_probe)

    models = fetch_acp_models("traex", cwd="/tmp/ghostap", probe_timeout=5.0)

    assert len(models) == 2
    assert models[0].name == "c_o_new_thinking"
    assert models[0].is_default is True


def test_probe_acp_models_initializes_lazy_providers_for_traex(
    monkeypatch,
    tmp_path,
):
    """Traex model probing must not depend on some earlier code path having
    already initialized the ACP provider registry."""
    from src.acp.helper import probe_acp_models
    from src.acp.providers import _reset_providers_for_testing

    _reset_providers_for_testing()

    class FakeModels:
        available_models = []
        current_model_id = "c_o_new_thinking/medium"

    class FakeOption:
        def __init__(self, name, value):
            self.name = name
            self.value = value
            self.description = ""

    class FakeRoot:
        id = "model"
        category = "model"
        current_value = "c_o_new_thinking"
        options = [
            FakeOption("Test-O-New-Thinking", "c_o_new_thinking"),
            FakeOption("GPT-5.5", "gpt-5.5"),
        ]

    class FakeConfigOption:
        root = FakeRoot()

    class FakeReasoningRoot:
        id = "reasoning_effort"
        category = "thought_level"
        current_value = "high"
        options = [
            FakeOption("Low", "low"),
            FakeOption("High", "high"),
        ]

    class FakeReasoningConfigOption:
        root = FakeReasoningRoot()

    class FakeResponse:
        session_id = "session-1"
        models = FakeModels()
        config_options = [FakeConfigOption(), FakeReasoningConfigOption()]

    class FakeConn:
        async def initialize(self, protocol_version):
            return None

        async def new_session(self, cwd):
            return FakeResponse()

        async def set_config_option(self, **_kwargs):
            return FakeResponse()

    class FakeSpawn:
        async def __aenter__(self):
            return FakeConn(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def fake_spawn_agent_process(_client, cmd, *args, **_kwargs):
        assert cmd == "traex"
        assert args == ("acp", "serve")
        return FakeSpawn()

    monkeypatch.setattr("src.acp.helper.spawn_agent_process", fake_spawn_agent_process)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    models = asyncio.run(probe_acp_models("traex", cwd="/tmp/ghostap"))

    assert [m.name for m in models] == ["c_o_new_thinking", "gpt-5.5"]
    assert [m.name for m in models if m.is_default] == ["c_o_new_thinking"]
    assert all(
        {variant.profile for variant in model.selection_variants} == {"standard"}
        for model in models
    )


def test_probe_codex_models_reads_exact_efforts_for_each_model(monkeypatch):
    from src.acp.helper import probe_acp_models

    class FakeOption:
        def __init__(self, name, value):
            self.name = name
            self.value = value
            self.description = ""

    class FakeRoot:
        def __init__(self, option_id, category, current_value, options):
            self.id = option_id
            self.category = category
            self.current_value = current_value
            self.options = options

    class FakeConfigOption:
        def __init__(self, root):
            self.root = root

    model_options = [
        FakeOption("GPT-5.5", "gpt-5.5"),
        FakeOption("GPT-5.6-Sol", "gpt-5.6-sol"),
    ]

    def response(model, efforts):
        return SimpleNamespace(
            models=None,
            config_options=[
                FakeConfigOption(
                    FakeRoot("model", "model", model, model_options)
                ),
                FakeConfigOption(
                    FakeRoot(
                        "reasoning_effort",
                        "thought_level",
                        "high",
                        [FakeOption(value, value) for value in efforts],
                    )
                ),
            ],
        )

    initial_response = response(
        "gpt-5.6-sol",
        ["low", "medium", "high", "xhigh", "max", "ultra"],
    )
    per_model = {
        "gpt-5.5": response(
            "gpt-5.5",
            ["low", "medium", "high", "xhigh"],
        ),
        "gpt-5.6-sol": initial_response,
    }

    class FakeConn:
        current_model = "gpt-5.6-sol"

        async def initialize(self, protocol_version):
            return None

        async def new_session(self, cwd):
            return initial_response

        async def set_config_option(self, *, session_id, config_id, value):
            assert session_id == "session-1"
            if config_id == "reasoning_effort":
                assert value == "high"
            else:
                assert config_id == "model"
                self.current_model = value
            return per_model[self.current_model]

    initial_response.session_id = "session-1"
    for item in per_model.values():
        item.session_id = "session-1"

    class FakeSpawn:
        async def __aenter__(self):
            return FakeConn(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "src.acp.helper.spawn_agent_process",
        lambda *_args, **_kwargs: FakeSpawn(),
    )

    models = asyncio.run(probe_acp_models("codex", cwd="/tmp/ghostap"))

    assert [model.name for model in models] == ["gpt-5.5", "gpt-5.6-sol"]
    assert models[0].reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert models[1].reasoning_efforts == (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    )
    assert models[1].adapted_reasoning_effort == "high"
    assert models[1].is_default is True


def test_probe_codex_models_resets_initial_selection_before_each_capability_read(
    monkeypatch,
):
    from src.acp.helper import probe_acp_models

    class FakeOption:
        def __init__(self, value):
            self.name = value
            self.value = value
            self.description = ""

    class FakeRoot:
        def __init__(self, option_id, category, current_value, options):
            self.id = option_id
            self.category = category
            self.current_value = current_value
            self.options = [FakeOption(value) for value in options]

    class FakeConfigOption:
        def __init__(self, root):
            self.root = root

    def response(model, effort, supported):
        return SimpleNamespace(
            session_id="session-1",
            config_options=[
                FakeConfigOption(
                    FakeRoot(
                        "model",
                        "model",
                        model,
                        ["model-a", "model-b"],
                    )
                ),
                FakeConfigOption(
                    FakeRoot(
                        "reasoning_effort",
                        "thought_level",
                        effort,
                        supported,
                    )
                ),
            ],
        )

    class FakeConn:
        current_model = "model-a"
        current_effort = "ultra"
        calls = []

        async def initialize(self, protocol_version):
            return None

        async def new_session(self, cwd):
            return response(
                self.current_model,
                self.current_effort,
                ["high", "ultra"],
            )

        async def set_config_option(self, *, session_id, config_id, value):
            self.calls.append((config_id, value))
            if config_id == "model":
                self.current_model = value
                supported = (
                    ["high", "ultra"]
                    if value == "model-a"
                    else ["medium", "high"]
                )
                if self.current_effort not in supported:
                    self.current_effort = "medium"
            else:
                self.current_effort = value
                supported = (
                    ["high", "ultra"]
                    if self.current_model == "model-a"
                    else ["medium", "high"]
                )
            return response(
                self.current_model,
                self.current_effort,
                supported,
            )

    connection = FakeConn()

    class FakeSpawn:
        async def __aenter__(self):
            return connection, object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "src.acp.helper.spawn_agent_process",
        lambda *_args, **_kwargs: FakeSpawn(),
    )

    models = asyncio.run(probe_acp_models("codex", cwd="/tmp/ghostap"))

    assert [model.adapted_reasoning_effort for model in models] == [
        "ultra",
        "medium",
    ]
    assert connection.calls == [
        ("model", "model-a"),
        ("reasoning_effort", "ultra"),
        ("model", "model-a"),
        ("model", "model-a"),
        ("reasoning_effort", "ultra"),
        ("model", "model-b"),
    ]


def test_probe_codex_models_filters_unknown_future_efforts(monkeypatch):
    from src.acp.helper import probe_acp_models

    class FakeOption:
        def __init__(self, value):
            self.name = value
            self.value = value
            self.description = ""

    class FakeRoot:
        def __init__(self, option_id, category, current_value, options):
            self.id = option_id
            self.category = category
            self.current_value = current_value
            self.options = [FakeOption(value) for value in options]

    class FakeConfigOption:
        def __init__(self, root):
            self.root = root

    def response():
        return SimpleNamespace(
            session_id="session-1",
            config_options=[
                FakeConfigOption(
                    FakeRoot("model", "model", "model-a", ["model-a"])
                ),
                FakeConfigOption(
                    FakeRoot(
                        "reasoning_effort",
                        "thought_level",
                        "high",
                        ["high", "future-effort"],
                    )
                ),
            ],
        )

    class FakeConn:
        async def initialize(self, protocol_version):
            return None

        async def new_session(self, cwd):
            return response()

        async def set_config_option(self, **_kwargs):
            return response()

    class FakeSpawn:
        async def __aenter__(self):
            return FakeConn(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "src.acp.helper.spawn_agent_process",
        lambda *_args, **_kwargs: FakeSpawn(),
    )

    models = asyncio.run(probe_acp_models("codex", cwd="/tmp/ghostap"))

    assert models[0].reasoning_efforts == ("high",)


# ---------------------------------------------------------------------------
# Traex model/Profile/Effort capability extraction
# ---------------------------------------------------------------------------


def _traex_select_root(config_id, current_value, values):
    root = SimpleNamespace(
        id=config_id,
        category="model" if config_id == "model" else "thought_level",
        current_value=current_value,
        options=[SimpleNamespace(name=value, value=value) for value in values],
    )
    return SimpleNamespace(root=root)


def _traex_config_response(
    *,
    session_id="session-1",
    current_model="c_o_new_thinking",
    efforts=("low", "medium", "high", "max"),
):
    return SimpleNamespace(
        session_id=session_id,
        config_options=[
            _traex_select_root(
                "model",
                current_model,
                ("c_o_new_thinking",),
            ),
            _traex_select_root("reasoning_effort", efforts[-1], efforts),
        ],
    )


def _write_traex_capability_cache(tmp_path):
    levels = [{"effort": value} for value in ("low", "medium", "high", "max")]
    cache = tmp_path / "models_cache.json"
    cache.write_text(
        json.dumps({
            "models": [
                {
                    "slug": "Test-O-New-Thinking",
                    "config_name": "c_o_new_thinking",
                    "default_reasoning_level": "high",
                    "supported_reasoning_levels": levels,
                    "business_metadata": {
                        "variants": {
                            "standard_key": "c_o_new_thinking__dev",
                            "standard_default_reasoning_level": "high",
                            "standard_supported_reasoning_levels": levels,
                            "max_key": "c_o_new_thinking__max",
                            "max_default_reasoning_level": "high",
                            "max_supported_reasoning_levels": levels,
                        }
                    },
                }
            ]
        }),
        encoding="utf-8",
    )
    return cache


class _FakeTraexCapabilityConnection:
    def __init__(self):
        self.writes = []

    async def set_config_option(self, *, session_id, config_id, value):
        self.writes.append((config_id, value))
        assert session_id == "session-1"
        assert config_id == "model"
        return _traex_config_response()


def test_extract_traex_capabilities_builds_profile_effort_variants(tmp_path):
    from src.acp.helper import _extract_traex_model_capabilities

    conn = _FakeTraexCapabilityConnection()
    models = asyncio.run(
        _extract_traex_model_capabilities(
            conn,
            _traex_config_response(),
            current_model=None,
            metadata_path=_write_traex_capability_cache(tmp_path),
        )
    )

    assert len(models) == 1
    target = models[0]
    assert target.name == "c_o_new_thinking"
    assert {variant.name for variant in target.selection_variants} == {
        f"c_o_new_thinking/{profile}/{effort}"
        for profile in ("standard", "max")
        for effort in ("low", "medium", "high", "max")
    }
    assert {
        (variant.profile, variant.effort)
        for variant in target.selection_variants
        if variant.is_variant_default
    } == {("standard", "max"), ("max", "max")}
    assert conn.writes == [
        ("model", "c_o_new_thinking"),
        ("model", "c_o_new_thinking__max"),
    ]


def test_extract_traex_capabilities_without_cache_only_uses_standard_profile(
    tmp_path,
):
    from src.acp.helper import _extract_traex_model_capabilities

    conn = _FakeTraexCapabilityConnection()
    models = asyncio.run(
        _extract_traex_model_capabilities(
            conn,
            _traex_config_response(),
            current_model=None,
            metadata_path=tmp_path / "missing.json",
        )
    )

    assert models
    assert {variant.profile for variant in models[0].selection_variants} == {
        "standard"
    }
    assert conn.writes == [("model", "c_o_new_thinking")]


def test_mark_default_matches_traex_composite_variant():
    from src.acp.helper import _mark_default
    from src.ttadk.models import ACPModelOption, ACPModelVariantOption

    models = [
        ACPModelOption(
            name="c_o_new_thinking",
            selection_variants=(
                ACPModelVariantOption(
                    name="c_o_new_thinking/max/max",
                    profile="max",
                    effort="max",
                ),
            ),
        )
    ]

    marked = _mark_default(models, "c_o_new_thinking/max/max")

    assert marked[0].is_default is True


def test_invalidate_acp_model_cache_clears_entries_and_increments_generation():
    from src.acp import helper
    from src.ttadk.models import ACPModelOption

    key = helper._probe_key("traex", "/repo")
    with helper._acp_probe_cache_lock:
        helper._acp_probe_cache[key] = (0.0, [ACPModelOption(name="old")])
        helper._acp_neg_cache[key] = 0.0
    before = helper.get_acp_model_cache_generation("traex", "/repo")

    helper.invalidate_acp_model_cache("traex", "/repo")

    assert key not in helper._acp_probe_cache
    assert key not in helper._acp_neg_cache
    assert helper.get_acp_model_cache_generation("traex", "/repo") == before + 1
