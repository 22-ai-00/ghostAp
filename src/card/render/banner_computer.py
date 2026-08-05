"""Unified banner computation for sticky_head."""

from __future__ import annotations

from src.card.state.models import CardMetadata
from src.card.state.runtime_stats import RuntimeStats


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as `XmYYs` or `Ys`."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    minutes = s // 60
    rem = s % 60
    return f"{minutes}m{rem}s"


def compute_banner(metadata: CardMetadata, runtime: RuntimeStats) -> str:
    """Build engine phase banner text.

    Plain programming cards already show project/tool/model in the header and
    footer; rendering a generic ``Coco · 进行中 · 0s`` line in the body repeats
    that context and wastes mobile vertical space. Keep sticky phase banners for
    engines where phase/cycle/subagent context is meaningful on repeated pages.
    """
    if not metadata.engine_type:
        return ""
    emoji = metadata.mode_emoji or "🤖"
    mode = metadata.mode_name or "Programming"
    phase = _format_phase(metadata, runtime)
    elapsed = format_elapsed(runtime.elapsed_seconds)
    return f"{emoji} {mode} · {phase} · {elapsed}"


def _format_phase(metadata: CardMetadata, runtime: RuntimeStats) -> str:
    """Engine-specific phase string."""
    engine = metadata.engine_type
    if engine == "deep":
        if runtime.deep_phase == "analyzing":
            return "分析中"
        return "执行中"
    if engine == "spec":
        cycle = runtime.spec_cycle if runtime.spec_cycle is not None else "?"
        persp = runtime.spec_perspective or "—"
        return f"cycle {cycle}/{persp}"
    return "进行中"
