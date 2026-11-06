"""Spec shared utilities.

目标：把可复用的解析/校验逻辑下沉到 utils 层。

- Review 解析（严格/宽松两条路径，不含 LLM 兜底）
- Criteria 评估的正则 patterns
- JSON 产物提取（```json fenced block``` 优先）
- Spec/Plan 结构化产物校验（用于在 UI 中显式提示降级）
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..engine_base import PerspectiveReview, ReviewPerspective

# ---------------------------------------------------------------------------
# Criteria patterns
# ---------------------------------------------------------------------------


CRITERIA_PATTERNS: list[re.Pattern] = [re.compile(rf"CRITERIA_{i}\s*:\s*(PASS|FAIL)") for i in range(1, 101)]


# ---------------------------------------------------------------------------
# JSON extraction / normalization
# ---------------------------------------------------------------------------


def normalize_list(items) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for x in items:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def extract_json_blob(text: str) -> Optional[str]:
    """Extract a JSON object string from text.

    优先匹配 ```json fenced block```，否则退化为第一个 '{' 到最后一个 '}'。
    """
    if not text:
        return None
    raw = text.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            p = part.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                return p
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return None


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------


def validate_spec_artifact_dict(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["Spec JSON 不是对象"]

    required_list_fields = [
        "goals",
        "functional_spec",
        "non_functional_requirements",
        "acceptance_criteria",
        "out_of_scope",
        "risks",
        "clarification_questions",
        "decisions",
    ]
    for k in required_list_fields:
        v = data.get(k)
        if not isinstance(v, list):
            errors.append(f"Spec 字段 `{k}` 不是数组")

    # soft constraints
    ac = data.get("acceptance_criteria")
    if isinstance(ac, list) and len([x for x in ac if str(x).strip()]) == 0:
        errors.append("Spec 字段 `acceptance_criteria` 为空（应提供可验收条件）")

    return errors


def validate_plan_artifact_dict(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["Plan JSON 不是对象"]

    if not isinstance(data.get("architecture"), str):
        errors.append("Plan 字段 `architecture` 不是字符串")

    required_list_fields = [
        "tech_stack",
        "steps",
        "file_changes",
        "test_plan",
        "risks",
    ]
    for k in required_list_fields:
        v = data.get(k)
        if not isinstance(v, list):
            errors.append(f"Plan 字段 `{k}` 不是数组")

    steps = data.get("steps")
    if isinstance(steps, list) and len([x for x in steps if str(x).strip()]) == 0:
        errors.append("Plan 字段 `steps` 为空（应给出可执行步骤）")

    return errors


# ---------------------------------------------------------------------------
# Review parsing (strict/tolerant, no LLM)
# ---------------------------------------------------------------------------


REVIEW_SECTION_PATTERN = re.compile(
    r"\[(\w+)\]\s*\n\s*(PASS|FAIL)\b(.*?)(?=\[(?:ARCHITECT|PRODUCT|USER|TESTER|DESIGNER)\]|\Z)",
    re.DOTALL,
)


REVIEW_HEADER_EN_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?im)^\s*(?:#+\s*)?\[\s*(ARCHITECT|PRODUCT|USER|TESTER|DESIGNER)\s*\]\s*(?:[:：]?\s*(.*))?$"),
    re.compile(
        r"(?im)^\s*(?:#+\s*)?\*{1,2}\[\s*(ARCHITECT|PRODUCT|USER|TESTER|DESIGNER)\s*\]\*{1,2}\s*(?:[:：]?\s*(.*))?$"
    ),
    re.compile(r"(?im)^\s*(?:#+\s*)?\*{1,2}(ARCHITECT|PRODUCT|USER|TESTER|DESIGNER)\*{1,2}\s*(?:[:：]?\s*(.*))?$"),
    re.compile(r"(?im)^\s*#+\s*(ARCHITECT|PRODUCT|USER|TESTER|DESIGNER)\s*(?:[:：]?\s*(.*))?$"),
    re.compile(r"(?im)^\s*(ARCHITECT|PRODUCT|USER|TESTER|DESIGNER)\s*[:：]\s*(.*)$"),
]

REVIEW_HEADER_ZH_PATTERN = re.compile(
    r"(?im)^\s*(?:#+\s*)?(?:\d+[.、)]\s*)?(?:[-*]\s*)?"
    r"(?:🏗️|📦|👤|🧪|🎨)?\s*"
    r"\*{0,2}"
    r"(架构师|产品经理|用户|测试|设计师)"
    r"\*{0,2}"
    r"(?:审查|评审|视角)?"
    r"\s*(?:[:：]\s*(.*))?$",
)

PERSPECTIVE_ZH_MAP: dict[str, ReviewPerspective] = {
    "架构师": ReviewPerspective.ARCHITECT,
    "产品经理": ReviewPerspective.PRODUCT,
    "用户": ReviewPerspective.USER,
    "测试": ReviewPerspective.TESTER,
    "设计师": ReviewPerspective.DESIGNER,
}

PERSPECTIVE_TAG_MAP: dict[str, ReviewPerspective] = {
    "ARCHITECT": ReviewPerspective.ARCHITECT,
    "PRODUCT": ReviewPerspective.PRODUCT,
    "USER": ReviewPerspective.USER,
    "TESTER": ReviewPerspective.TESTER,
    "DESIGNER": ReviewPerspective.DESIGNER,
}


def normalize_review_verdict(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()
    if "PASS" in t:
        return "PASS"
    if "FAIL" in t:
        return "FAIL"
    if "通过" in text and "不通过" not in text:
        return "PASS"
    if "不通过" in text or "未通过" in text or "失败" in text:
        return "FAIL"
    return None


BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\d+、)\s*(.+?)\s*$")


def extract_suggestions_from_body(body: str, limit: int = 10) -> list[str]:
    suggestions: list[str] = []
    if not body:
        return suggestions
    for line in body.split("\n"):
        m = BULLET_PATTERN.match(line)
        if not m:
            continue
        s = (m.group(1) or "").strip()
        if s:
            suggestions.append(s)
        if len(suggestions) >= limit:
            break
    return suggestions


def split_review_sections(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    normalized = text.replace("\r\n", "\n")
    # hits: (start, end, tag, header_tail)
    hits: list[tuple[int, int, str, str]] = []
    seen_positions: set[int] = set()
    for pat in REVIEW_HEADER_EN_PATTERNS:
        for m in pat.finditer(normalized):
            if m.start() not in seen_positions:
                seen_positions.add(m.start())
                tail = (m.group(2) or "").strip()
                hits.append((m.start(), m.end(), m.group(1).upper(), tail))
    for m in REVIEW_HEADER_ZH_PATTERN.finditer(normalized):
        zh = (m.group(1) or "").strip()
        p = PERSPECTIVE_ZH_MAP.get(zh)
        if not p:
            continue
        tail = (m.group(2) or "").strip()
        hits.append((m.start(), m.end(), p.name, tail))

    if not hits:
        return []
    hits.sort(key=lambda x: x[0])
    sections: list[tuple[str, str]] = []
    for i, (_start, end, tag, header_tail) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(normalized)
        block = normalized[end:next_start].strip("\n")
        # Preserve same-line verdicts like "[ARCHITECT] FAIL" by injecting
        # the header tail as the first line of the section body.
        if header_tail:
            combined = header_tail if not block else (header_tail + "\n" + block)
        else:
            combined = block
        sections.append((tag, combined))
    return sections


# ---------------------------------------------------------------------------
# Review parsing — Level 2.5: loose (keyword-pair / JSON / table)
# ---------------------------------------------------------------------------


# Map of keywords → ReviewPerspective for loose matching
_LOOSE_PERSPECTIVE_KEYWORDS: dict[str, ReviewPerspective] = {
    "architect": ReviewPerspective.ARCHITECT,
    "架构师": ReviewPerspective.ARCHITECT,
    "架构": ReviewPerspective.ARCHITECT,
    "product": ReviewPerspective.PRODUCT,
    "产品经理": ReviewPerspective.PRODUCT,
    "产品": ReviewPerspective.PRODUCT,
    "user": ReviewPerspective.USER,
    "用户": ReviewPerspective.USER,
    "tester": ReviewPerspective.TESTER,
    "测试": ReviewPerspective.TESTER,
    "designer": ReviewPerspective.DESIGNER,
    "设计师": ReviewPerspective.DESIGNER,
    "设计": ReviewPerspective.DESIGNER,
}

_LOOSE_PASS_KEYWORDS = {"pass", "通过"}
_LOOSE_FAIL_KEYWORDS = {"fail", "不通过", "未通过", "失败"}

_LOOSE_LINE_PATTERN = re.compile(r"(?i)\b(" + "|".join(re.escape(k) for k in _LOOSE_PERSPECTIVE_KEYWORDS) + r")\b")


def _match_verdict_in_text(text: str) -> Optional[bool]:
    """Check if text contains a PASS or FAIL verdict. Returns True=PASS, False=FAIL, None=no match."""
    lower = text.lower()
    for kw in _LOOSE_FAIL_KEYWORDS:
        if kw in lower:
            return False
    for kw in _LOOSE_PASS_KEYWORDS:
        if kw in lower:
            return True
    return None


def _try_parse_json_reviews(text: str) -> list[PerspectiveReview]:
    """Try to parse reviews from JSON array format."""
    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    reviews: list[PerspectiveReview] = []
    found: set[ReviewPerspective] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        # Try to find perspective from any key
        perspective = None
        for key in ("perspective", "role", "视角", "角色"):
            val = str(item.get(key, "")).strip().lower()
            perspective = _LOOSE_PERSPECTIVE_KEYWORDS.get(val)
            if perspective:
                break
        if not perspective or perspective in found:
            continue
        found.add(perspective)

        # Try to find verdict
        verdict_val = None
        for key in ("verdict", "result", "结果", "passed"):
            v = str(item.get(key, "")).strip()
            if v:
                verdict_val = _match_verdict_in_text(v)
                if verdict_val is not None:
                    break
        passed = verdict_val if verdict_val is not None else True

        suggestions_raw = item.get("suggestions", [])
        if not isinstance(suggestions_raw, list):
            suggestions_raw = []
        suggestions = [str(s) for s in suggestions_raw if s] if not passed else []

        reviews.append(
            PerspectiveReview(
                perspective=perspective,
                passed=passed,
                suggestions=suggestions,
                summary=f"{'通过' if passed else f'{len(suggestions)}条建议'}",
            )
        )
    return reviews


def parse_review_output_loose(text: str, iteration: int) -> list[PerspectiveReview]:
    """Level 2.5 loose parsing — keyword-pair matching, JSON array, table formats.

    Sits between strict/tolerant and LLM fallback. Does not require section headers.
    Scans for perspective keywords co-occurring with PASS/FAIL verdicts.
    """
    if not text:
        return []
    raw = text.replace("\r\n", "\n")

    # Strategy 1: Try JSON array parsing (agent may output raw JSON)
    json_reviews = _try_parse_json_reviews(raw)
    if len(json_reviews) >= 2:
        return json_reviews

    # Strategy 2: Line-by-line keyword-pair scanning
    lines = raw.split("\n")
    reviews: list[PerspectiveReview] = []
    found: set[ReviewPerspective] = set()

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        # Find perspective in current line
        for keyword, perspective in _LOOSE_PERSPECTIVE_KEYWORDS.items():
            if keyword not in line_lower:
                continue
            if perspective in found:
                continue

            # Look for verdict in same line
            verdict = _match_verdict_in_text(line)

            # If not found in same line, check next 2 lines
            if verdict is None:
                for j in range(1, min(3, len(lines) - i)):
                    verdict = _match_verdict_in_text(lines[i + j])
                    if verdict is not None:
                        break

            if verdict is None:
                continue

            found.add(perspective)
            # Collect suggestions from subsequent bullet lines
            suggestions: list[str] = []
            if not verdict:
                for j in range(1, min(8, len(lines) - i)):
                    next_line = lines[i + j].strip()
                    m = BULLET_PATTERN.match(next_line)
                    if m:
                        s = (m.group(1) or "").strip()
                        if s and _match_verdict_in_text(s) is None:
                            suggestions.append(s)
                    elif next_line and _LOOSE_LINE_PATTERN.search(next_line):
                        break  # Next perspective section
                    elif not next_line:
                        continue

            reviews.append(
                PerspectiveReview(
                    perspective=perspective,
                    passed=verdict,
                    suggestions=suggestions,
                    summary=f"{'通过' if verdict else f'{len(suggestions)}条建议'}",
                )
            )
            break  # Move to next line after finding a match

    # Strategy 3: Table format (| 架构师 | PASS |)
    if not reviews:
        table_pattern = re.compile(
            r"\|\s*([^|]+?)\s*\|\s*(PASS|FAIL|通过|不通过|未通过|失败)\s*\|",
            re.IGNORECASE,
        )
        for m in table_pattern.finditer(raw):
            cell = m.group(1).strip().lower()
            perspective = _LOOSE_PERSPECTIVE_KEYWORDS.get(cell)
            if not perspective or perspective in found:
                continue
            found.add(perspective)
            verdict_text = m.group(2).strip()
            passed = _match_verdict_in_text(verdict_text)
            if passed is None:
                passed = True
            reviews.append(
                PerspectiveReview(
                    perspective=perspective,
                    passed=passed,
                    suggestions=[],
                    summary=f"{'通过' if passed else '有建议'}",
                )
            )

    return reviews
