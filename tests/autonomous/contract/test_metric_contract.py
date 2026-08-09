"""Contract tests for acceptance metrics calculations."""

from __future__ import annotations

import math

import pytest

from src.autonomous.acceptance.metrics import (
    percentiles,
    wilson_confidence_interval,
    zero_event_upper_bound,
)


def test_percentiles_empty() -> None:
    result = percentiles([])
    assert result.sample_count == 0
    assert result.p50 == 0.0


def test_percentiles_single_value() -> None:
    result = percentiles([42.0])
    assert result.p50 == 42.0
    assert result.p95 == 42.0
    assert result.p99 == 42.0


def test_percentiles_sorted() -> None:
    values = list(range(1, 101))  # 1..100
    result = percentiles([float(v) for v in values])
    assert result.sample_count == 100
    assert result.p50 == pytest.approx(50.5, abs=0.5)
    assert result.p95 == pytest.approx(95.05, abs=0.5)
    assert result.p99 == pytest.approx(99.01, abs=0.5)


def test_wilson_confidence_interval_basic() -> None:
    result = wilson_confidence_interval(90, 100)
    assert 0.82 < result.lower < 0.92
    assert 0.93 < result.upper < 0.97
    assert result.n == 100
    assert result.successes == 90


def test_wilson_zero_trials() -> None:
    result = wilson_confidence_interval(0, 0)
    assert result.lower == 0.0
    assert result.upper == 1.0


def test_wilson_perfect_score() -> None:
    result = wilson_confidence_interval(100, 100)
    assert result.lower > 0.95
    assert result.upper > 0.99


def test_zero_event_bound_finite() -> None:
    result = zero_event_upper_bound(100.0, confidence=0.95)
    expected = -math.log(0.05) / 100.0
    assert result.upper_bound == pytest.approx(expected, rel=1e-6)


def test_zero_event_bound_zero_time() -> None:
    result = zero_event_upper_bound(0.0)
    assert result.upper_bound == float("inf")


def test_zero_event_bound_high_confidence() -> None:
    r95 = zero_event_upper_bound(1000.0, confidence=0.95)
    r99 = zero_event_upper_bound(1000.0, confidence=0.99)
    assert r99.upper_bound > r95.upper_bound
