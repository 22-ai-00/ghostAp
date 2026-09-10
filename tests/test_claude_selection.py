"""Unit tests for the Claude ``base[1m]/effort`` selection codec."""

from __future__ import annotations

import pytest

from src.acp.claude_selection import (
    CLAUDE_REASONING_EFFORTS,
    compose_claude_model_selection,
    split_claude_model_selection,
)


def test_effort_catalog_matches_cli_help() -> None:
    # Locked to the values reported by `claude --effort` / `claude-w --effort`.
    assert CLAUDE_REASONING_EFFORTS == ("low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize("effort", CLAUDE_REASONING_EFFORTS)
def test_split_roundtrip(effort: str) -> None:
    base, parsed_effort = split_claude_model_selection(f"claude-sonnet-4-5/{effort}")
    assert base == "claude-sonnet-4-5"
    assert parsed_effort == effort


def test_split_keeps_1m_suffix_on_base() -> None:
    base, effort = split_claude_model_selection("claude-opus-4-8[1m]/high")
    assert base == "claude-opus-4-8[1m]"
    assert effort == "high"


def test_split_no_effort_returns_whole_selection() -> None:
    assert split_claude_model_selection("claude-sonnet-4-5") == (
        "claude-sonnet-4-5",
        None,
    )
    assert split_claude_model_selection("claude-opus-4-8[1m]") == (
        "claude-opus-4-8[1m]",
        None,
    )


def test_split_unknown_effort_token_is_not_stripped() -> None:
    # Provider-qualified ids containing "/" stay intact.
    base, effort = split_claude_model_selection("vendor/claude-sonnet-4-5")
    assert base == "vendor/claude-sonnet-4-5"
    assert effort is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_split_empty(value: str | None) -> None:
    assert split_claude_model_selection(value) == (None, None)


def test_compose_with_effort() -> None:
    assert compose_claude_model_selection("claude-sonnet-4-5", "xhigh") == "claude-sonnet-4-5/xhigh"


def test_compose_with_1m_base_keeps_suffix_before_effort() -> None:
    assert compose_claude_model_selection("claude-opus-4-8[1m]", "max") == "claude-opus-4-8[1m]/max"


def test_compose_without_effort_returns_base() -> None:
    assert compose_claude_model_selection("claude-sonnet-4-5", None) == ("claude-sonnet-4-5")
    assert compose_claude_model_selection("claude-sonnet-4-5", "") == ("claude-sonnet-4-5")


def test_compose_rejects_unknown_effort() -> None:
    # An unrecognised level must never be silently embedded — the CLI
    # rejects unknown --effort values and the session would fail to start.
    assert compose_claude_model_selection("claude-sonnet-4-5", "ultra") == ("claude-sonnet-4-5")


def test_compose_empty_base() -> None:
    assert compose_claude_model_selection("", "high") == ""
