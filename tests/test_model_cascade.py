from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.acp.helper import (
    build_traex_model_options,
    discover_codex_model_options,
)
from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.acp.traex_selection import (
    TraexModelMetadata,
    TraexProfileMetadata,
)
from src.card.render.model_cascade import (
    available_model_efforts,
    available_model_names,
    available_model_profiles,
    compose_model_selection,
    parse_model_selection,
    resolve_model_cascade,
    validate_model_selection,
)


def _variant(
    name: str,
    model: str,
    *,
    profile: str | None = None,
    effort: str | None = None,
    is_default: bool = False,
) -> ACPModelSelectionVariant:
    return ACPModelSelectionVariant(
        name=name,
        model=model,
        profile=profile,
        effort=effort,
        is_default=is_default,
    )


def _select_option(
    option_id: str,
    current: str,
    values: tuple[tuple[str, str, str], ...],
    *,
    category: str,
) -> SimpleNamespace:
    root = SimpleNamespace(
        id=option_id,
        category=category,
        current_value=current,
        options=[
            SimpleNamespace(value=value, name=label, description=description)
            for value, label, description in values
        ],
    )
    return SimpleNamespace(root=root)


def _codex_response(
    *,
    current_model: str,
    efforts: tuple[str, ...],
    models: tuple[str, ...] = ("gpt-5.6-sol", "gpt-5.5"),
) -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        config_options=[
            _select_option(
                "model",
                current_model,
                tuple((model, model.upper(), f"Description for {model}") for model in models),
                category="model",
            ),
            _select_option(
                "reasoning_effort",
                efforts[0] if efforts else "",
                tuple((effort, effort.title(), effort) for effort in efforts),
                category="thought_level",
            ),
        ],
    )


def test_plain_models_are_a_single_exact_selection_dimension() -> None:
    models = [
        ACPModelOption(name="plain-a", is_default=True),
        ACPModelOption(name="plain-b"),
    ]

    state = resolve_model_cascade(models)

    assert available_model_names(models) == ("plain-a", "plain-b")
    assert available_model_profiles(models, "plain-a") == ()
    assert available_model_efforts(models, "plain-a") == ()
    assert state.selected_model == "plain-a"
    assert state.selection == "plain-a"
    assert compose_model_selection(models, model="plain-b") == "plain-b"
    assert compose_model_selection(models, model="plain-b", effort="high") is None
    assert validate_model_selection(models, "plain-b") is True
    assert validate_model_selection(models, "plain-b/high") is False


def test_codex_downstream_efforts_stay_scoped_to_each_model() -> None:
    models = [
        ACPModelOption(
            name="gpt-5.6-sol",
            is_default=True,
            reasoning_efforts=("high", "max", "ultra"),
            default_reasoning_effort="high",
            selection_variants=(
                _variant("gpt-5.6-sol/high", "gpt-5.6-sol", effort="high", is_default=True),
                _variant("gpt-5.6-sol/max", "gpt-5.6-sol", effort="max"),
                _variant("gpt-5.6-sol/ultra", "gpt-5.6-sol", effort="ultra"),
            ),
        ),
        ACPModelOption(
            name="gpt-5.5",
            reasoning_efforts=("low", "high"),
            default_reasoning_effort="high",
            selection_variants=(
                _variant("gpt-5.5/low", "gpt-5.5", effort="low"),
                _variant("gpt-5.5/high", "gpt-5.5", effort="high", is_default=True),
            ),
        ),
    ]

    state = resolve_model_cascade(models, selected_model="gpt-5.5")

    assert available_model_efforts(models, "gpt-5.6-sol") == ("high", "max", "ultra")
    assert available_model_efforts(models, "gpt-5.5") == ("low", "high")
    assert state.selected_model == "gpt-5.5"
    assert state.selected_effort == "high"
    assert state.selection == "gpt-5.5/high"
    assert compose_model_selection(models, model="gpt-5.5", effort="max") is None
    assert validate_model_selection(models, "gpt-5.5/max") is False


def test_traex_profiles_and_efforts_are_strictly_cascaded() -> None:
    models = [
        ACPModelOption(
            name="c_o_new",
            is_default=True,
            selection_variants=(
                _variant(
                    "c_o_new/standard/medium",
                    "c_o_new",
                    profile="standard",
                    effort="medium",
                    is_default=True,
                ),
                _variant(
                    "c_o_new/standard/high",
                    "c_o_new",
                    profile="standard",
                    effort="high",
                ),
                _variant(
                    "c_o_new/max/xhigh",
                    "c_o_new",
                    profile="max",
                    effort="xhigh",
                    is_default=True,
                ),
            ),
        )
    ]

    state = resolve_model_cascade(
        models,
        current_model="c_o_new/max/xhigh",
    )

    assert available_model_profiles(models, "c_o_new") == ("standard", "max")
    assert available_model_efforts(models, "c_o_new", "standard") == ("medium", "high")
    assert available_model_efforts(models, "c_o_new", "max") == ("xhigh",)
    assert state.selected_profile == "max"
    assert state.selected_effort == "xhigh"
    assert state.selection == "c_o_new/max/xhigh"
    assert compose_model_selection(
        models,
        model="c_o_new",
        profile="standard",
        effort="xhigh",
    ) is None


def test_saved_selection_is_parsed_only_when_the_exact_variant_is_available() -> None:
    models = [
        ACPModelOption(
            name="gpt-5.6-sol",
            selection_variants=(
                _variant("gpt-5.6-sol/high", "gpt-5.6-sol", effort="high"),
            ),
        )
    ]

    parsed = parse_model_selection(models, "gpt-5.6-sol/high")

    assert parsed is not None
    assert (parsed.model, parsed.profile, parsed.effort, parsed.value) == (
        "gpt-5.6-sol",
        None,
        "high",
        "gpt-5.6-sol/high",
    )
    assert parse_model_selection(models, "gpt-5.6-sol/ultra") is None


def test_codex_discovery_uses_each_model_update_response_as_its_authority() -> None:
    initial = _codex_response(
        current_model="gpt-5.6-sol",
        efforts=("high", "max", "ultra"),
    )

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def set_config_option(self, *, config_id: str, session_id: str, value: str):
            self.calls.append((config_id, session_id, value))
            return _codex_response(current_model=value, efforts=("low", "high"))

    connection = Connection()

    models = asyncio.run(discover_codex_model_options(connection, initial))

    assert connection.calls == [("model", "session-1", "gpt-5.5")]
    assert models[0].reasoning_efforts == ("high", "max", "ultra")
    assert models[1].reasoning_efforts == ("low", "high")
    assert [variant.name for variant in models[1].selection_variants] == [
        "gpt-5.5/low",
        "gpt-5.5/high",
    ]
    assert "gpt-5.5/max" not in {
        variant.name for model in models for variant in model.selection_variants
    }


def test_codex_discovery_does_not_borrow_efforts_when_one_model_probe_fails() -> None:
    initial = _codex_response(
        current_model="gpt-5.6-sol",
        efforts=("high", "max"),
    )

    class Connection:
        async def set_config_option(self, **_kwargs):
            raise RuntimeError("probe failed")

    models = asyncio.run(discover_codex_model_options(Connection(), initial))

    assert [variant.name for variant in models[0].selection_variants] == [
        "gpt-5.6-sol/high",
        "gpt-5.6-sol/max",
    ]
    assert [variant.name for variant in models[1].selection_variants] == ["gpt-5.5"]
    assert models[1].reasoning_efforts == ()


def test_traex_capabilities_are_the_intersection_of_live_models_and_metadata() -> None:
    live = [ACPModelOption(name="c_o_new", is_default=True)]
    metadata = (
        TraexModelMetadata(
            config_name="c_o_new",
            slug="Test-O-New",
            profiles=(
                TraexProfileMetadata(
                    profile="standard",
                    backend_model_value="c_o_new",
                    reasoning_efforts=("medium", "high"),
                    default_effort="medium",
                ),
                TraexProfileMetadata(
                    profile="max",
                    backend_model_value="c_o_new__max",
                    reasoning_efforts=("xhigh",),
                    default_effort="xhigh",
                ),
            ),
        ),
        TraexModelMetadata(
            config_name="metadata_only",
            slug="Metadata-Only",
            profiles=(
                TraexProfileMetadata(
                    profile="standard",
                    backend_model_value="metadata_only",
                ),
            ),
        ),
    )

    models = build_traex_model_options(live, metadata)

    assert [model.name for model in models] == ["c_o_new"]
    assert [variant.name for variant in models[0].selection_variants] == [
        "c_o_new/standard/medium",
        "c_o_new/standard/high",
        "c_o_new/max/xhigh",
    ]
    assert models[0].selection_variants[0].is_default is True
    assert models[0].selection_variants[2].is_default is True


def test_traex_model_without_metadata_keeps_only_the_safe_plain_selection() -> None:
    models = build_traex_model_options(
        [ACPModelOption(name="live-only", is_default=True)],
        (),
    )

    assert [variant.name for variant in models[0].selection_variants] == ["live-only"]
    assert models[0].selection_variants[0].profile == "standard"
    assert models[0].selection_variants[0].effort is None
