"""banner_computer unit tests."""
from __future__ import annotations

from src.card.render.banner_computer import compute_banner, format_elapsed
from src.card.state.models import CardMetadata
from src.card.state.runtime_stats import RuntimeStats


def test_format_elapsed_under_one_minute():
    assert format_elapsed(45.0) == "45s"


def test_format_elapsed_minute_seconds():
    assert format_elapsed(83.0) == "1m23s"


def test_format_elapsed_zero():
    assert format_elapsed(0.0) == "0s"


def test_banner_deep_executing():
    md = CardMetadata(mode_name="Deep", mode_emoji="🧠", engine_type="deep")
    rs = RuntimeStats(elapsed_seconds=83.0, deep_phase="executing")
    assert compute_banner(md, rs) == "🧠 Deep · 执行中 · 1m23s"


def test_banner_deep_analyzing():
    md = CardMetadata(mode_name="Deep", mode_emoji="🧠", engine_type="deep")
    rs = RuntimeStats(elapsed_seconds=10.0, deep_phase="analyzing")
    assert compute_banner(md, rs) == "🧠 Deep · 分析中 · 10s"


def test_banner_spec_cycle_perspective():
    md = CardMetadata(mode_name="Spec", mode_emoji="📐", engine_type="spec")
    rs = RuntimeStats(elapsed_seconds=484.0, spec_cycle=2, spec_perspective="code")
    assert compute_banner(md, rs) == "📐 Spec · cycle 2/code · 8m4s"


def test_banner_omits_plain_programming_context():
    md = CardMetadata(mode_name=None, mode_emoji=None, engine_type=None)  # type: ignore[arg-type]
    rs = RuntimeStats(elapsed_seconds=5.0)
    assert compute_banner(md, rs) == ""
