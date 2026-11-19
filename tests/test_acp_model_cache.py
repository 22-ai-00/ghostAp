from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    NewSessionResponse,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
)

from src.acp import helper
from src.acp.options import ACPModelOption
from src.acp.transport import LateFrameTolerantMessageQueue


@pytest.fixture(autouse=True)
def _isolated_model_cache():
    with helper._cache_lock:
        helper._probe_cache.clear()
        helper._negative_cache.clear()
        helper._inflight.clear()
        helper._cache_generation.clear()
    yield
    with helper._cache_lock:
        helper._probe_cache.clear()
        helper._negative_cache.clear()
        helper._inflight.clear()
        helper._cache_generation.clear()


def test_positive_model_catalog_is_reused_for_thirty_minutes(
    monkeypatch,
    tmp_path,
) -> None:
    now = [1_000.0]
    probe = MagicMock(return_value=[ACPModelOption(name="model-a")])
    monkeypatch.setattr(helper.time, "time", lambda: now[0])
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )

    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "model-a"
    now[0] += 1_799.9
    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "model-a"
    now[0] += 0.2
    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "model-a"

    assert probe.call_count == 2


def test_failed_model_catalog_is_not_reprobed_for_five_minutes(
    monkeypatch,
    tmp_path,
) -> None:
    now = [2_000.0]
    probe = MagicMock(return_value=[])
    monkeypatch.setattr(helper.time, "time", lambda: now[0])
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )

    assert helper.fetch_acp_models("traex", str(tmp_path)) == []
    now[0] += 299.9
    assert helper.fetch_acp_models("traex", str(tmp_path)) == []
    now[0] += 0.2
    assert helper.fetch_acp_models("traex", str(tmp_path)) == []

    assert probe.call_count == 2


def test_exact_invalidation_reprobes_only_the_requested_catalog(
    monkeypatch,
    tmp_path,
) -> None:
    probe = MagicMock(
        side_effect=lambda tool, *_args: [ACPModelOption(name=f"{tool}-model")]
    )
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )

    first_cwd = str(tmp_path / "first")
    second_cwd = str(tmp_path / "second")
    helper.fetch_acp_models("traex", first_cwd)
    helper.fetch_acp_models("traex", second_cwd)
    helper.fetch_acp_models("codex", first_cwd)
    helper.invalidate_acp_model_cache("traex", first_cwd)
    helper.fetch_acp_models("traex", first_cwd)
    helper.fetch_acp_models("traex", second_cwd)
    helper.fetch_acp_models("codex", first_cwd)

    assert [(call.args[0], call.args[1]) for call in probe.call_args_list] == [
        ("traex", first_cwd),
        ("traex", second_cwd),
        ("codex", first_cwd),
        ("traex", first_cwd),
    ]


def test_generic_model_probe_uses_late_frame_tolerant_queue(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class Connection:
        async def initialize(self, protocol_version: int = 1) -> None:
            assert protocol_version == 1

        async def new_session(self, cwd: str):
            return SimpleNamespace(models=None)

    @asynccontextmanager
    async def spawn(*_args, **kwargs):
        captured.update(kwargs)
        yield Connection(), object()

    provider = SimpleNamespace(get_serve_command=lambda _model: ("agent", []))
    monkeypatch.setattr(helper, "get_providers", lambda: {"aiden": provider})
    monkeypatch.setattr(helper, "spawn_agent_process", spawn)

    assert asyncio.run(helper._probe_acp_models("aiden", str(tmp_path))) == []
    queue = captured.get("queue")
    assert isinstance(queue, LateFrameTolerantMessageQueue)

    async def publish_after_close() -> None:
        await queue.close()
        await queue.publish(object())

    asyncio.run(publish_after_close())


def test_traex_probe_prefers_config_catalog_over_active_model_sentinel(
    monkeypatch,
    tmp_path,
) -> None:
    class Connection:
        async def initialize(self, protocol_version: int = 1) -> None:
            assert protocol_version == 1

        async def new_session(self, cwd: str):
            assert cwd == str(tmp_path)
            return SimpleNamespace(
                models=SimpleNamespace(
                    current_model_id="TraeX active model",
                    available_models=(
                        SimpleNamespace(
                            model_id="TraeX active model",
                            name="TraeX active model",
                            description="Current model only",
                        ),
                    ),
                ),
                config_options=(
                    SimpleNamespace(
                        root=SimpleNamespace(
                            id="model",
                            category="model",
                            current_value="model-b",
                            options=(
                                SimpleNamespace(
                                    value="model-a",
                                    name="Model A",
                                    description="First model",
                                ),
                                SimpleNamespace(
                                    value="model-b",
                                    name="Model B",
                                    description="Second model",
                                ),
                            ),
                        )
                    ),
                ),
            )

    @asynccontextmanager
    async def spawn(*_args, **_kwargs):
        yield Connection(), object()

    provider = SimpleNamespace(get_serve_command=lambda _model: ("traex", []))
    monkeypatch.setattr(helper, "get_providers", lambda: {"traex": provider})
    monkeypatch.setattr(helper, "spawn_agent_process", spawn)
    monkeypatch.setattr(helper, "load_traex_model_metadata", lambda: ())

    models = asyncio.run(helper._probe_acp_models("traex", str(tmp_path)))

    assert [model.name for model in models] == ["model-a", "model-b"]
    assert [model.is_default for model in models] == [False, True]
    assert "TraeX active model" not in {model.name for model in models}


def test_codex_discovery_passes_local_cli_to_official_adapter(monkeypatch, tmp_path):
    captured = {}

    class Connection:
        async def initialize(self, **kwargs):
            pass

        async def new_session(self, **kwargs):
            return SimpleNamespace(config_options=())

    @asynccontextmanager
    async def spawn(*args, **kwargs):
        captured.update(kwargs)
        yield Connection(), object()

    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.setattr("src.acp.transport.shutil.which", lambda name, **kwargs: "/local/codex")
    monkeypatch.setattr(helper, "get_providers", lambda: {"codex": SimpleNamespace(
        get_serve_command=lambda model: ("npx", ["--yes", "@agentclientprotocol/codex-acp@1.2.0"]),
    )})
    monkeypatch.setattr(helper, "spawn_agent_process", spawn)
    assert asyncio.run(helper._probe_acp_models("codex", str(tmp_path))) == []
    assert captured["env"]["CODEX_PATH"] == "/local/codex"


def test_traex_retired_default_bootstraps_live_catalog_with_profiles_and_efforts(
    monkeypatch, tmp_path,
) -> None:
    from src.acp.traex_selection import TraexModelMetadata, TraexProfileMetadata
    from src.card.render.model_cascade import resolve_model_cascade

    metadata = (
        TraexModelMetadata("first", "First", (TraexProfileMetadata("standard", "first"),)),
        TraexModelMetadata("model-b", "Model B", (
            TraexProfileMetadata("standard", "model-b", ("low", "high"), "high"),
            TraexProfileMetadata("max", "model-b__max", ("high", "max"), "max"),
        )),
    )
    starts = []

    def command(model):
        starts.append(model)
        return "traex", []

    class Connection:
        async def initialize(self, **kwargs):
            pass

        async def new_session(self, cwd):
            if starts[-1] is None:
                raise RequestError(-32603, "Internal error", 'model `retired` metadata could not be resolved')
            return SimpleNamespace(config_options=(SimpleNamespace(
                id="model", current_value="model-b", options=(
                    SimpleNamespace(value="model-b", name="Model B"),
                ),
            ),))

    @asynccontextmanager
    async def spawn(*args, **kwargs):
        yield Connection(), object()

    monkeypatch.setattr(helper, "get_providers", lambda: {"traex": SimpleNamespace(get_serve_command=command)})
    monkeypatch.setattr(helper, "spawn_agent_process", spawn)
    monkeypatch.setattr(helper, "load_traex_model_metadata", lambda: metadata)

    models = helper.fetch_acp_models("traex", str(tmp_path), "model-b/max/high")
    assert starts == [None, "model-b"]
    assert [model.name for model in models] == ["model-b"]  # Never expose cache-only models.
    state = resolve_model_cascade(models, current_model="model-b/max/high")
    assert state.profiles == ("standard", "max")
    assert state.efforts == ("high", "max")
    assert state.selection == "model-b/max/high"


@pytest.mark.parametrize("tool,detail,expected_attempts", [
    ("traex", "metadata could not be resolved", 2),
    ("traex", "authentication failed", 1),
    ("codex", "metadata could not be resolved", 1),
])
def test_model_probe_retry_is_bounded_and_only_for_traex_retired_metadata(
    monkeypatch, tmp_path, tool, detail, expected_attempts,
) -> None:
    from src.acp.traex_selection import TraexModelMetadata

    probe = AsyncMock(side_effect=RequestError(-32603, "Internal error", detail))
    monkeypatch.setattr(helper, "_probe_acp_models_once", probe)
    monkeypatch.setattr(helper, "load_traex_model_metadata", lambda: (
        TraexModelMetadata("available", "Available", ()),
    ))
    with pytest.raises(RequestError):
        asyncio.run(helper._probe_acp_models(tool, str(tmp_path)))
    assert probe.await_count == expected_attempts
    if expected_attempts == 2:
        assert probe.call_args.args == (tool, str(tmp_path), "available")


def test_invalidation_fences_an_older_inflight_probe_result(
    monkeypatch,
    tmp_path,
) -> None:
    old_started = threading.Event()
    release_old = threading.Event()
    probe_calls = 0
    probe_lock = threading.Lock()

    def probe(_tool, *_args):
        nonlocal probe_calls
        with probe_lock:
            probe_calls += 1
            call_number = probe_calls
        if call_number == 1:
            old_started.set()
            assert release_old.wait(timeout=2)
            return [ACPModelOption(name="old-model")]
        return [ACPModelOption(name="new-model")]

    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )
    old_result: list[list[ACPModelOption]] = []
    worker = threading.Thread(
        target=lambda: old_result.append(
            helper.fetch_acp_models("traex", str(tmp_path))
        )
    )
    worker.start()
    assert old_started.wait(timeout=1)

    helper.invalidate_acp_model_cache("traex", str(tmp_path))
    fresh = helper.fetch_acp_models("traex", str(tmp_path))
    release_old.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert fresh[0].name == "new-model"
    assert old_result[0][0].name == "old-model"
    assert helper.fetch_acp_models("traex", str(tmp_path))[0].name == "new-model"
    assert probe_calls == 2


def test_cache_fill_between_miss_and_flight_registration_does_not_reprobe(
    monkeypatch,
    tmp_path,
) -> None:
    delayed_miss = threading.Event()
    release_delayed = threading.Event()
    original_cached = helper._cached
    delayed_once = False

    def cached(key, tool):
        nonlocal delayed_once
        result = original_cached(key, tool)
        if threading.current_thread().name == "delayed-model-fetch" and not delayed_once:
            delayed_once = True
            delayed_miss.set()
            assert release_delayed.wait(timeout=2)
        return result

    probe = MagicMock(return_value=[ACPModelOption(name="model-a")])
    monkeypatch.setattr(helper, "_cached", cached)
    monkeypatch.setattr(helper, "_probe_blocking", probe)
    monkeypatch.setattr(
        "src.agent_session.backend_resolver.is_cli_backend",
        lambda _tool: False,
    )
    delayed_result: list[list[ACPModelOption]] = []
    delayed_worker = threading.Thread(
        name="delayed-model-fetch",
        target=lambda: delayed_result.append(
            helper.fetch_acp_models("traex", str(tmp_path))
        ),
    )
    delayed_worker.start()
    assert delayed_miss.wait(timeout=1)

    leading_result = helper.fetch_acp_models("traex", str(tmp_path))
    release_delayed.set()
    delayed_worker.join(timeout=2)

    assert not delayed_worker.is_alive()
    assert leading_result[0].name == "model-a"
    assert delayed_result[0][0].name == "model-a"
    assert probe.call_count == 1


def test_grouped_select_options_flatten_across_codex_and_config_models() -> None:
    def make_options(*, include_efforts: bool, current_model: str) -> list:
        options: list = [
            SessionConfigOptionBoolean(
                id="model",
                name="Bool model flag",
                type="boolean",
                category="model",
                current_value=True,
            ),
            SessionConfigOptionSelect(
                id="ignored_dict_cat",
                name="Ignored",
                type="select",
                category={"id": "model"},
                current_value="should-not-match",
                options=[SessionConfigSelectOption(value="should-not-match", name="Ignored")],
            ),
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                type="select",
                category="model",
                current_value=current_model,
                options=[
                    SessionConfigSelectGroup(
                        group="empty",
                        name="Empty Group",
                        options=[],
                    ),
                    SessionConfigSelectGroup(
                        group="flagship",
                        name="Flagship",
                        options=[
                            SessionConfigSelectOption(value="gpt-5.4", name="GPT-5.4"),
                            SessionConfigSelectOption(value="gpt-5.4-mini", name="GPT-5.4 Mini"),
                        ],
                    ),
                ],
            ),
            SessionConfigOptionBoolean(
                id="reasoning_effort",
                name="Enable effort",
                type="boolean",
                category="thought_level",
                current_value=False,
            ),
            SessionConfigOptionSelect(
                id="other_dict_cat",
                name="Dict category effort",
                type="select",
                category={"id": "thought_level"},
                current_value="xhigh",
                options=[SessionConfigSelectOption(value="xhigh", name="XHigh")],
            ),
        ]
        if include_efforts:
            options.append(
                SessionConfigOptionSelect(
                    id="reasoning_effort",
                    name="Reasoning Effort",
                    type="select",
                    category="thought_level",
                    current_value="high",
                    options=[
                        SessionConfigSelectGroup(group="empty_effort", name="Empty", options=[]),
                        SessionConfigSelectGroup(
                            group="tiers",
                            name="Tiers",
                            options=[
                                SessionConfigSelectOption(value="medium", name="Medium"),
                                SessionConfigSelectOption(value="high", name="High"),
                            ],
                        ),
                    ],
                )
            )
        return options

    initial = NewSessionResponse(
        session_id="sess-codex-grouped",
        config_options=make_options(include_efforts=True, current_model="gpt-5.4"),
    )

    class Connection:
        async def set_config_option(self, *, config_id: str, session_id: str, value: str):
            assert config_id == "model"
            assert session_id == "sess-codex-grouped"
            return SetSessionConfigOptionResponse(
                config_options=make_options(include_efforts=False, current_model=value),
            )

    # _config_models reuses _config_option + _flatten_config_options
    flat_models = helper._config_models(initial, "")
    assert [m.name for m in flat_models] == ["gpt-5.4", "gpt-5.4-mini"]
    assert [m.is_default for m in flat_models] == [True, False]

    # discover_codex_model_options flattens grouped models and grouped efforts,
    # and NEVER backfills static efforts when a model reports no effort capability.
    discovered = asyncio.run(helper.discover_codex_model_options(Connection(), initial))
    assert [m.name for m in discovered] == ["gpt-5.4", "gpt-5.4-mini"]
    assert discovered[0].reasoning_efforts == ("medium", "high")
    assert discovered[0].default_reasoning_effort == "high"
    assert [v.name for v in discovered[0].selection_variants] == [
        "gpt-5.4/medium",
        "gpt-5.4/high",
    ]
    assert discovered[1].reasoning_efforts == ()
    assert discovered[1].default_reasoning_effort is None
    assert [v.name for v in discovered[1].selection_variants] == ["gpt-5.4-mini"]


@pytest.mark.parametrize(
    "bad_options",
    [
        "not-a-list",
        b"bytes-not-a-list",
        {"value": "mapping"},
        42,
        [SimpleNamespace(group="bad-nested", options="bad-str")],
        [SimpleNamespace(group="bad-nested-map", options={"k": "v"})],
        [SimpleNamespace(group="bad-nested-int", options=99)],
    ],
)
def test_flatten_config_options_ignores_malformed_non_iterable_options(bad_options) -> None:
    root = SimpleNamespace(options=bad_options)
    assert helper._flatten_config_options(root) == []
