"""Tests for the synthetic effort dimension served to claude/claude-w CLI.

CLI backends expose no ACP model catalog; ``fetch_acp_models`` must still
return a ``base[1m]/effort`` matrix so the model cascade card can render the
effort dropdown on first paint.
"""

from __future__ import annotations

from src.acp import helper
from src.acp.claude_selection import (
    CLAUDE_DEFAULT_EFFORT_TOKEN,
    CLAUDE_DEFAULT_MODEL_TOKEN,
    CLAUDE_REASONING_EFFORTS,
)
from src.card.render.model_cascade import (
    compose_model_selection,
    resolve_model_cascade,
)


def _variant_names(option) -> list[str]:
    return [v.name for v in option.selection_variants]


def test_cli_catalog_defaults_to_pseudo_model_with_six_variants() -> None:
    options = helper.fetch_acp_models("claude", None, current_model=None)

    assert len(options) == 1
    only = options[0]
    assert only.name == CLAUDE_DEFAULT_MODEL_TOKEN
    # default effort (omit flag) + five real levels.
    assert _variant_names(only) == [
        CLAUDE_DEFAULT_MODEL_TOKEN,
        *[f"{CLAUDE_DEFAULT_MODEL_TOKEN}/{level}" for level in CLAUDE_REASONING_EFFORTS],
    ]
    assert only.reasoning_efforts == CLAUDE_REASONING_EFFORTS
    # With no saved selection the plain default is the active variant.
    assert only.is_default is True
    assert only.selection_variants[0].is_default is True
    assert only.selection_variants[0].effort == CLAUDE_DEFAULT_EFFORT_TOKEN


def test_cli_catalog_roundtrips_saved_base_and_effort() -> None:
    options = helper.fetch_acp_models("claude", None, current_model="claude-sonnet-4-5/high")

    names = {option.name for option in options}
    assert names == {CLAUDE_DEFAULT_MODEL_TOKEN, "claude-sonnet-4-5"}

    saved = next(o for o in options if o.name == "claude-sonnet-4-5")
    high = next(v for v in saved.selection_variants if v.name == "claude-sonnet-4-5/high")
    assert high.is_default is True
    assert high.effort == "high"
    assert saved.is_default is True
    # The pseudo-model must not be flagged default once a base is saved.
    pseudo = next(o for o in options if o.name == CLAUDE_DEFAULT_MODEL_TOKEN)
    assert pseudo.is_default is False


def test_cli_catalog_roundtrips_saved_1m_composite() -> None:
    options = helper.fetch_acp_models("claude_w", None, current_model="claude-opus-4-8[1m]/max")

    saved = next(o for o in options if o.name == "claude-opus-4-8[1m]")
    max_variant = next(v for v in saved.selection_variants if v.name == "claude-opus-4-8[1m]/max")
    assert max_variant.is_default is True


def test_cli_catalog_plain_saved_base_marks_default_effort_variant() -> None:
    options = helper.fetch_acp_models("claude", None, current_model="claude-sonnet-4-5")

    saved = next(o for o in options if o.name == "claude-sonnet-4-5")
    plain = next(v for v in saved.selection_variants if v.name == "claude-sonnet-4-5")
    assert plain.is_default is True
    assert plain.effort == CLAUDE_DEFAULT_EFFORT_TOKEN


def test_cli_catalog_serves_claude_w_identically() -> None:
    options = helper.fetch_acp_models("claude_w", None, current_model=None)
    assert len(options) == 1
    assert options[0].name == CLAUDE_DEFAULT_MODEL_TOKEN
    assert options[0].reasoning_efforts == CLAUDE_REASONING_EFFORTS


def test_cascade_first_paint_renders_effort_dropdown_without_profile() -> None:
    options = helper.fetch_acp_models("claude", None, current_model=None)
    state = resolve_model_cascade(options, current_model=None)

    assert state.model_names == (CLAUDE_DEFAULT_MODEL_TOKEN,)
    # No profile dimension for Claude — only model + effort dropdowns.
    assert state.profiles == ()
    assert state.selected_profile is None
    assert state.efforts == (
        CLAUDE_DEFAULT_EFFORT_TOKEN,
        *CLAUDE_REASONING_EFFORTS,
    )
    # Default effort collapses the composite back to the bare pseudo-base,
    # which the CLI bridge interprets as "omit both flags".
    assert state.selection == CLAUDE_DEFAULT_MODEL_TOKEN


def test_cascade_picking_effort_composes_composite_value() -> None:
    options = helper.fetch_acp_models("claude", None, current_model=None)
    state = resolve_model_cascade(
        options,
        current_model=None,
        selected_model=CLAUDE_DEFAULT_MODEL_TOKEN,
        selected_effort="xhigh",
    )
    assert state.selection == "default/xhigh"


def test_cascade_explicit_effort_roundtrips_as_current() -> None:
    options = helper.fetch_acp_models("claude", None, current_model="claude-sonnet-4-5/high")
    state = resolve_model_cascade(options, current_model="claude-sonnet-4-5/high")
    assert state.selected_model == "claude-sonnet-4-5"
    assert state.selected_effort == "high"
    assert state.selection == "claude-sonnet-4-5/high"
    # And composing from the resolved triple reproduces the stored value.
    assert (
        compose_model_selection(
            options,
            model=state.selected_model,
            profile=state.selected_profile,
            effort=state.selected_effort,
        )
        == "claude-sonnet-4-5/high"
    )
