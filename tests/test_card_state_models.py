"""Immutability contract for card state transitions."""

import dataclasses
from dataclasses import replace

import pytest

from src.card.state.models import CardState


def test_card_state_changes_use_replace() -> None:
    state = CardState()

    with pytest.raises(dataclasses.FrozenInstanceError):
        state.version = 99  # type: ignore[misc]

    updated = replace(state, version=1)
    assert (state.version, updated.version) == (0, 1)
