"""Shared fail-closed budget policy for employee context layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import (
    ContextLayer,
    ContextMessage,
    ContextUnavailableError,
    ContextUnavailableReason,
    ThreadContextConfig,
    TrimmingRecord,
)


@dataclass(frozen=True, slots=True)
class BudgetedContextLayers:
    l1_summary: str
    l2_summary: str
    total_chars: int
    total_tokens: int
    trimming_trace: tuple[TrimmingRecord, ...]


def apply_context_budget(
    *,
    config: ThreadContextConfig,
    thread: list[ContextMessage],
    group: list[ContextMessage],
    l1_summary: str,
    l2_summary: str,
    reserve: int,
) -> BudgetedContextLayers:
    """Trim low-priority layers while treating the complete topic as atomic."""

    trace: list[TrimmingRecord] = []

    def totals() -> tuple[int, int]:
        chars = (
            sum(len(message.text) for message in thread)
            + sum(len(message.text) for message in group)
            + len(l1_summary)
            + len(l2_summary)
        )
        return chars, math.ceil(chars * config.tokens_per_char) + reserve

    def over_budget() -> bool:
        chars, tokens = totals()
        return chars > config.max_context_chars or tokens > config.max_context_tokens

    while over_budget():
        if l2_summary:
            removed = len(l2_summary)
            l2_summary = ""
            _record_trim(trace, ContextLayer.L2_GROUP, 0, removed)
            continue
        if group:
            removed = group.pop(0)
            _record_trim(trace, ContextLayer.GROUP_RECENT, 1, len(removed.text))
            continue
        if l1_summary:
            removed = len(l1_summary)
            l1_summary = ""
            _record_trim(trace, ContextLayer.L1_MEMORY, 0, removed)
            continue
        raise ContextUnavailableError(ContextUnavailableReason.BUDGET)

    total_chars, total_tokens = totals()
    return BudgetedContextLayers(
        l1_summary=l1_summary,
        l2_summary=l2_summary,
        total_chars=total_chars,
        total_tokens=total_tokens,
        trimming_trace=tuple(trace),
    )


def _record_trim(
    trace: list[TrimmingRecord],
    layer: ContextLayer,
    removed_messages: int,
    removed_chars: int,
) -> None:
    if trace and trace[-1].layer is layer:
        previous = trace[-1]
        trace[-1] = TrimmingRecord(
            layer,
            previous.removed_messages + removed_messages,
            previous.removed_chars + removed_chars,
        )
        return
    trace.append(TrimmingRecord(layer, removed_messages, removed_chars))


__all__ = ["BudgetedContextLayers", "apply_context_budget"]
