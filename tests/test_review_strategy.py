"""Tests for ReviewStrategy registry + NoReview / MultiPerspective wrappers."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from src.engine_base import ReviewPerspective, ReviewResult
from src.spec_engine.review import ReviewCircuitState
from src.spec_engine.review_strategy import (
    AdaptiveRoleReviewStrategy,
    MultiPerspectiveStrategy,
    NoReviewStrategy,
    ReviewContext,
    select_review_strategy,
)


def _make_ctx(**overrides):
    base = dict(
        cycle=1,
        session=None,
        settings=types.SimpleNamespace(
            spec_review_failure_circuit_enabled=False,
            spec_review_failure_max_consecutive=3,
            spec_review_failure_cooldown_cycles=2,
        ),
        project=None,
        send_prompt_with_retry_fn=MagicMock(),
        build_review_exception_diagnostics_fn=MagicMock(return_value={}),
        circuit=ReviewCircuitState(),
        on_review_done=None,
    )
    base.update(overrides)
    return ReviewContext(**base)


def test_strategy_selection_contract():
    cases = (
        (types.SimpleNamespace(), AdaptiveRoleReviewStrategy),
        (types.SimpleNamespace(spec_review_strategy="multi_perspective"), MultiPerspectiveStrategy),
        (types.SimpleNamespace(spec_review_strategy="none"), NoReviewStrategy),
        (types.SimpleNamespace(spec_review_strategy="xxx"), MultiPerspectiveStrategy),
    )
    for settings, expected_type in cases:
        assert isinstance(select_review_strategy(settings), expected_type), settings


def test_no_review_returns_empty():
    called = []
    ctx = _make_ctx(on_review_done=lambda c, r: called.append((c, r)))
    r = NoReviewStrategy().run(ctx)
    assert isinstance(r, ReviewResult)
    assert r.iteration == 1
    assert len(called) == 1
    assert called[0][0] == 1


def test_multi_perspective_no_session_returns_empty():
    # session=None → conduct_review early-exits with empty ReviewResult
    ctx = _make_ctx(session=None)
    r = MultiPerspectiveStrategy().run(ctx)
    assert isinstance(r, ReviewResult)
    assert r.iteration == 1
    # All perspectives default to pass=True / no reviews in empty mode
    assert r.reviews == []


def test_multi_perspective_circuit_open_skips(monkeypatch):
    circuit = ReviewCircuitState(
        review_failure_consecutive=3,
        review_circuit_open_until_cycle=5,
    )
    settings = types.SimpleNamespace(
        spec_review_failure_circuit_enabled=True,
        spec_review_failure_max_consecutive=3,
        spec_review_failure_cooldown_cycles=2,
        review_circuit_lint_fallback_enabled=False,
    )
    ctx = _make_ctx(cycle=4, session=object(), settings=settings, circuit=circuit)
    r = MultiPerspectiveStrategy().run(ctx)
    assert isinstance(r, ReviewResult)
    # circuit-open path returns failed reviews for all perspectives
    assert len(r.reviews) == len(list(ReviewPerspective))
    assert all(not pr.passed for pr in r.reviews)
