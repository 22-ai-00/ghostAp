from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest


def _levels(*values: str) -> list[dict[str, str]]:
    return [{"effort": value} for value in values]


def _traex_cache_model() -> dict:
    levels = _levels("low", "medium", "high", "max")
    return {
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


def _write_cache(tmp_path, *models: dict):
    path = tmp_path / "models_cache.json"
    path.write_text(json.dumps({"models": list(models)}), encoding="utf-8")
    return path


def test_traex_selection_round_trips_explicit_profile_and_effort():
    from src.acp.traex_selection import (
        compose_traex_model_selection,
        split_traex_model_selection,
    )

    value = compose_traex_model_selection("c_o_new_thinking", "max", "max")

    assert value == "c_o_new_thinking/max/max"
    assert split_traex_model_selection(value) == (
        "c_o_new_thinking",
        "max",
        "max",
    )
    assert compose_traex_model_selection(
        "c_o_new_thinking", "standard", "high"
    ) == "c_o_new_thinking/standard/high"


def test_traex_selection_keeps_legacy_suffix_semantics():
    from src.acp.traex_selection import split_traex_model_selection

    assert split_traex_model_selection("c_o_new_thinking/high") == (
        "c_o_new_thinking",
        "standard",
        "high",
    )
    assert split_traex_model_selection("c_o_new_thinking/max") == (
        "c_o_new_thinking",
        "max",
        None,
    )
    assert split_traex_model_selection("c_o_new_thinking/max/high") == (
        "c_o_new_thinking",
        "max",
        "high",
    )


def test_runtime_selection_maps_max_profile_to_hidden_backend_key(tmp_path):
    from src.acp.traex_selection import resolve_traex_runtime_selection

    cache = _write_cache(tmp_path, _traex_cache_model())

    selection = resolve_traex_runtime_selection(
        "c_o_new_thinking/max/max",
        metadata_path=cache,
    )

    assert selection.model_id == "c_o_new_thinking"
    assert selection.backend_model_value == "c_o_new_thinking__max"
    assert selection.profile == "max"
    assert selection.effort == "max"


def test_runtime_selection_maps_standard_profile_to_bare_config_name(tmp_path):
    from src.acp.traex_selection import resolve_traex_runtime_selection

    cache = _write_cache(tmp_path, _traex_cache_model())

    selection = resolve_traex_runtime_selection(
        "Test-O-New-Thinking/standard/high",
        metadata_path=cache,
    )

    assert selection.model_id == "c_o_new_thinking"
    assert selection.backend_model_value == "c_o_new_thinking"
    assert selection.profile == "standard"
    assert selection.effort == "high"


def test_runtime_selection_rejects_unknown_profile_or_effort(tmp_path):
    from src.acp.traex_selection import resolve_traex_runtime_selection

    cache = _write_cache(tmp_path, _traex_cache_model())

    with pytest.raises(ValueError, match="profile"):
        resolve_traex_runtime_selection(
            "c_o_new_thinking/turbo/high",
            metadata_path=cache,
        )
    with pytest.raises(ValueError, match="effort"):
        resolve_traex_runtime_selection(
            "c_o_new_thinking/max/ultra",
            metadata_path=cache,
        )


def test_runtime_selection_without_cache_allows_only_standard_profile(tmp_path):
    from src.acp.traex_selection import resolve_traex_runtime_selection

    standard = resolve_traex_runtime_selection(
        "c_o_new_thinking/standard/high",
        metadata_path=tmp_path / "missing.json",
    )

    assert standard.backend_model_value == "c_o_new_thinking"
    assert standard.effort == "high"
    with pytest.raises(ValueError, match="profile"):
        resolve_traex_runtime_selection(
            "c_o_new_thinking/max/max",
            metadata_path=tmp_path / "missing.json",
        )


def test_expand_acp_model_options_emits_each_explicit_variant():
    from src.acp.options import ACPModelOption, ACPModelVariantOption
    from src.acp.traex_selection import expand_acp_model_options

    model = ACPModelOption(
        name="c_o_new_thinking",
        description="Test-O-New-Thinking",
        is_default=True,
        selection_variants=(
            ACPModelVariantOption(
                name="c_o_new_thinking/standard/high",
                profile="standard",
                effort="high",
                display_name="Test-O-New-Thinking · standard · high",
                is_variant_default=True,
            ),
            ACPModelVariantOption(
                name="c_o_new_thinking/max/max",
                profile="max",
                effort="max",
                display_name="Test-O-New-Thinking · max · max",
            ),
        ),
    )

    expanded = expand_acp_model_options([model])

    assert [item.name for item in expanded] == [
        "c_o_new_thinking/standard/high",
        "c_o_new_thinking/max/max",
    ]
    assert [item.is_default for item in expanded] == [True, False]
    assert expanded[0].description == "Test-O-New-Thinking · standard · high"


def test_acp_model_variant_options_are_immutable():
    from src.acp.options import ACPModelVariantOption

    variant = ACPModelVariantOption(
        name="c_o_new_thinking/max/max",
        profile="max",
        effort="max",
    )

    with pytest.raises(FrozenInstanceError):
        variant.effort = "high"


def test_expand_model_option_dicts_emits_explicit_variants():
    from src.acp.traex_selection import expand_model_option_dicts

    expanded = expand_model_option_dicts([
        {
            "name": "c_o_new_thinking",
            "display_name": "Test-O-New-Thinking",
            "description": "reasoning model",
            "is_default": True,
            "selection_variants": [
                {
                    "name": "c_o_new_thinking/standard/high",
                    "profile": "standard",
                    "effort": "high",
                    "display_name": "Test · standard · high",
                    "is_variant_default": True,
                },
                {
                    "name": "c_o_new_thinking/max/max",
                    "profile": "max",
                    "effort": "max",
                    "display_name": "Test · max · max",
                    "is_variant_default": False,
                },
            ],
        }
    ])

    assert [item["name"] for item in expanded] == [
        "c_o_new_thinking/standard/high",
        "c_o_new_thinking/max/max",
    ]
    assert [item["is_default"] for item in expanded] == [True, False]
