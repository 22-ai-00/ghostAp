"""Typed, deterministic Workflow result summaries for completion cards.

The completion card is an action surface, not the canonical result store.  This
module converts either the explicit ``card_summary`` contract or a small set of
legacy result keys into complete semantic items.  Unknown/raw content remains
available in the full Workflow report and is never recursively flattened here.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

_NEUTRAL_CONCLUSION = "任务已完成，完整结果见报告。"
_FAILED_FALLBACK_CONCLUSION = "任务未完成，请根据错误信息处理后重试。"
_CANCELLED_FALLBACK_CONCLUSION = "任务已取消，未生成完整最终结果。"
_TEXT_KEYS = ("text", "claim", "description", "issue", "summary", "name", "path")


class BriefVerdict(str, Enum):
    """Outcome presented on the completion card."""

    PASSED = "passed"
    NEEDS_ATTENTION = "needs_attention"
    FAILED = "failed"
    UNKNOWN = "unknown"


class BriefSeverity(str, Enum):
    """Stable severity ordering for key findings."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


_SEVERITY_ORDER = {
    BriefSeverity.HIGH: 0,
    BriefSeverity.MEDIUM: 1,
    BriefSeverity.LOW: 2,
    BriefSeverity.INFO: 3,
}


class BriefItem(BaseModel):
    """One complete semantic item rendered atomically on a card."""

    text: str
    severity: BriefSeverity = BriefSeverity.INFO
    status: str = "info"
    kind: str = "other"


class WorkflowResultBrief(BaseModel):
    """Normalized result surface consumed by the Workflow renderer."""

    verdict: BriefVerdict = BriefVerdict.UNKNOWN
    conclusion: str = _NEUTRAL_CONCLUSION
    findings: list[BriefItem] = Field(default_factory=list)
    verification: list[BriefItem] = Field(default_factory=list)
    deliverables: list[BriefItem] = Field(default_factory=list)
    next_steps: list[BriefItem] = Field(default_factory=list)
    omitted_counts: dict[str, int] = Field(default_factory=dict)
    source: Literal["contract", "legacy", "fallback"] = "fallback"


def build_result_brief(raw_result: str | None) -> WorkflowResultBrief:
    """Build a result brief without guessing the meaning of arbitrary output."""

    payload = _parse_object(raw_result)
    if payload is None:
        return WorkflowResultBrief()

    contract = payload.get("card_summary")
    if isinstance(contract, dict):
        return _brief_from_payload(contract, source="contract")

    brief = _brief_from_payload(payload, source="legacy")
    verification = payload.get("verification")
    if isinstance(verification, dict):
        approved = verification.get("approved")
        if approved is False:
            brief.verdict = BriefVerdict.NEEDS_ATTENTION
        elif approved is True and brief.verdict is BriefVerdict.UNKNOWN:
            brief.verdict = BriefVerdict.PASSED
        brief.findings.extend(_normalize_items(verification.get("issues")))
        brief.findings = _sort_findings(brief.findings)

    has_known_content = bool(
        brief.verdict is not BriefVerdict.UNKNOWN
        or brief.findings
        or brief.verification
        or brief.deliverables
        or brief.next_steps
        or brief.conclusion != _NEUTRAL_CONCLUSION
    )
    if not has_known_content:
        return WorkflowResultBrief()
    return brief


def fit_result_brief(
    brief: WorkflowResultBrief,
    *,
    max_text_bytes: int = 12_000,
    max_item_bytes: int = 900,
) -> WorkflowResultBrief:
    """Fit whole result items into a text budget without slicing strings."""

    omitted = dict(brief.omitted_counts)
    conclusion = brief.conclusion
    if _byte_len(conclusion) > max_item_bytes:
        conclusion = (
            _FAILED_FALLBACK_CONCLUSION
            if brief.verdict == BriefVerdict.FAILED
            else _NEUTRAL_CONCLUSION
        )
        omitted["conclusion"] = omitted.get("conclusion", 0) + 1

    used = _byte_len(conclusion)
    fitted: dict[str, list[BriefItem]] = {
        "findings": [],
        "verification": [],
        "deliverables": [],
        "next_steps": [],
    }
    for section in ("verification", "findings", "deliverables", "next_steps"):
        for item in getattr(brief, section):
            cost = _byte_len(item.text) + 32
            if cost > max_item_bytes or used + cost > max_text_bytes:
                omitted[section] = omitted.get(section, 0) + 1
                continue
            fitted[section].append(item)
            used += cost

    return brief.model_copy(
        update={
            "conclusion": conclusion,
            "findings": fitted["findings"],
            "verification": fitted["verification"],
            "deliverables": fitted["deliverables"],
            "next_steps": fitted["next_steps"],
            "omitted_counts": omitted,
        },
        deep=True,
    )


def _parse_object(raw_result: str | None) -> dict[str, Any] | None:
    text = str(raw_result or "").strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if not isinstance(value, dict):
        return ""
    for key in _TEXT_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _severity(value: Any) -> BriefSeverity:
    try:
        return BriefSeverity(str(value or "info").lower())
    except ValueError:
        return BriefSeverity.INFO


def _normalize_items(value: Any, *, kind: str = "other") -> list[BriefItem]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    items: list[BriefItem] = []
    for candidate in values:
        text = _first_text(candidate)
        if not text:
            continue
        metadata = candidate if isinstance(candidate, dict) else {}
        items.append(
            BriefItem(
                text=text,
                severity=_severity(metadata.get("severity")),
                status=str(metadata.get("status") or "info"),
                kind=str(metadata.get("type") or kind),
            )
        )
    return items


def _sort_findings(items: list[BriefItem]) -> list[BriefItem]:
    return sorted(items, key=lambda item: _SEVERITY_ORDER[item.severity])


def _derive_verdict(payload: dict[str, Any]) -> BriefVerdict:
    verdict_raw = payload.get("verdict")
    try:
        return BriefVerdict(str(verdict_raw))
    except ValueError:
        pass

    if payload.get("error"):
        return BriefVerdict.FAILED
    approved = payload.get("approved")
    if approved is False:
        return BriefVerdict.NEEDS_ATTENTION
    if approved is True:
        return BriefVerdict.PASSED
    return BriefVerdict.UNKNOWN


def _brief_from_payload(
    payload: dict[str, Any],
    *,
    source: Literal["contract", "legacy"],
) -> WorkflowResultBrief:
    verdict = _derive_verdict(payload)
    conclusion = ""
    for key in ("conclusion", "summary"):
        conclusion = _first_text(payload.get(key))
        if conclusion:
            break
    if not conclusion and verdict == BriefVerdict.FAILED:
        conclusion = _first_text(payload.get("error"))

    findings: list[BriefItem] = []
    for key in ("findings", "issues", "risks"):
        findings.extend(_normalize_items(payload.get(key)))

    verification = _normalize_items(payload.get("verification"), kind="verification")
    deliverables: list[BriefItem] = []
    for key in ("deliverables", "artifacts"):
        deliverables.extend(_normalize_items(payload.get(key), kind="artifact"))
    next_steps: list[BriefItem] = []
    for key in ("next_steps", "recommendations"):
        next_steps.extend(_normalize_items(payload.get(key), kind="next_step"))

    fallback_conclusion = (
        _FAILED_FALLBACK_CONCLUSION
        if verdict == BriefVerdict.FAILED
        else _NEUTRAL_CONCLUSION
    )
    return WorkflowResultBrief(
        verdict=verdict,
        conclusion=conclusion or fallback_conclusion,
        findings=_sort_findings(findings),
        verification=verification,
        deliverables=deliverables,
        next_steps=next_steps,
        source=source,
    )


def _byte_len(value: str) -> int:
    return len(value.encode("utf-8", errors="surrogatepass"))
