"""Artifact parsing and text utility functions for the Spec Engine."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

from ..utils.errors import get_error_detail
from .models import PlanArtifact, SpecArtifact, SpecTask
from .utils import (
    extract_json_blob,
    normalize_list,
    validate_plan_artifact_dict,
    validate_spec_artifact_dict,
)

if TYPE_CHECKING:
    from .models import SpecProject

_TASK_LINE_PATTERN = re.compile(
    r"^\s*(\d+)\s*[.、)]\s*(.+?)(?:\s*\(\s*(?:依赖|depends?)\s*[:：]?\s*(.*?)\s*\))?\s*$",
    re.IGNORECASE,
)


def safe_str(x: object) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def truncate_text(text: str, max_len: int, suffix: str = "...") -> str:
    try:
        s = str(text or "")
    except Exception:
        return ""
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if len(suffix) >= max_len:
        return s[:max_len]
    return s[: max_len - len(suffix)] + suffix


def parse_spec_artifact(text: str) -> tuple[Optional[SpecArtifact], list[str]]:
    """Parse spec JSON artifact.

    Returns: (artifact|None, validation_errors)
    """
    blob = extract_json_blob(text)
    if not blob:
        return None, ["未找到 ```json``` 规格产物；已降级为纯文本"]
    try:
        data = json.loads(blob)
    except Exception as e:
        return None, [f"规格 JSON 解析失败：{get_error_detail(e)}"]
    if not isinstance(data, dict):
        return None, ["规格 JSON 不是对象；已降级为纯文本"]

    errors = validate_spec_artifact_dict(data)
    artifact = SpecArtifact.from_dict(
        {
            "goals": normalize_list(data.get("goals")),
            "functional_spec": normalize_list(data.get("functional_spec")),
            "non_functional_requirements": normalize_list(data.get("non_functional_requirements")),
            "acceptance_criteria": normalize_list(data.get("acceptance_criteria")),
            "out_of_scope": normalize_list(data.get("out_of_scope")),
            "risks": normalize_list(data.get("risks")),
            "clarification_questions": normalize_list(data.get("clarification_questions")),
            "decisions": normalize_list(data.get("decisions")),
            "required_experts": data.get("required_experts") if isinstance(data.get("required_experts"), list) else [],
        }
    )
    return artifact, errors


def parse_plan_artifact(text: str) -> tuple[Optional[PlanArtifact], list[str]]:
    """Parse plan JSON artifact.

    Returns: (artifact|None, validation_errors)
    """
    blob = extract_json_blob(text)
    if not blob:
        return None, ["未找到 ```json``` 规划产物；已降级为纯文本"]
    try:
        data = json.loads(blob)
    except Exception as e:
        return None, [f"规划 JSON 解析失败：{get_error_detail(e)}"]
    if not isinstance(data, dict):
        return None, ["规划 JSON 不是对象；已降级为纯文本"]

    errors = validate_plan_artifact_dict(data)
    artifact = PlanArtifact.from_dict(
        {
            "architecture": data.get("architecture", ""),
            "tech_stack": normalize_list(data.get("tech_stack")),
            "steps": normalize_list(data.get("steps")),
            "file_changes": normalize_list(data.get("file_changes")),
            "test_plan": normalize_list(data.get("test_plan")),
            "risks": normalize_list(data.get("risks")),
        }
    )
    return artifact, errors


def merge_acceptance_criteria(project: SpecProject, new_criteria: list[str]) -> None:
    """Apply criteria from the spec artifact.

    spec-kit 的验收标准应来自 Spec 产物本身，因此在第一轮我们倾向于
    **用 Spec 的 acceptance_criteria 替换** 之前从用户输入中提取的
    fallback 列表（避免形成"双重标准"导致永远无法满足）。

    仅在以下情况下生效：
    - 还未有任何标准被满足（satisfied_count == 0）
    - 新标准不是明显的占位符（如 "CRITERIA_1" 这种标签）
    """
    if not project:
        return

    incoming = [c.strip() for c in (new_criteria or []) if c and c.strip()]
    if not incoming:
        return

    placeholder_pat = re.compile(r"^CRITERIA_\d+$", re.IGNORECASE)
    non_placeholder = [c for c in incoming if not placeholder_pat.match(c)]
    if not non_placeholder:
        return

    seen: set[str] = set()
    deduped: list[str] = []
    for c in non_placeholder:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)

    if project.criteria_tracker.satisfied_count == 0:
        project.acceptance_criteria = deduped
        project.criteria_tracker.init_criteria(deduped)


def parse_acceptance_criteria(text: str, decompose_fn: Optional[callable] = None) -> list[str]:
    """Extract acceptance criteria from user input."""
    lines = text.strip().split("\n")
    criteria = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            criteria.append(stripped[2:])
        elif stripped.startswith("[ ] ") or stripped.startswith("[x] "):
            criteria.append(stripped[4:])

    if not criteria and decompose_fn is not None:
        criteria = decompose_fn(text)

    if not criteria:
        criteria = [f"完成需求: {text[:100]}"]

    return criteria


def extract_criteria_from_llm_response(text: str) -> list[str]:
    criteria = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            criterion = stripped[2:].strip()
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".、):） ":
            criterion = stripped[2:].strip()
        elif len(stripped) > 3 and stripped[:2].isdigit() and stripped[2] in ".、):） ":
            criterion = stripped[3:].strip()
        else:
            continue
        if criterion:
            criteria.append(criterion)
    return criteria


def parse_tasks(text: str) -> list[SpecTask]:
    tasks = []
    for line in text.strip().split("\n"):
        m = _TASK_LINE_PATTERN.match(line)
        if not m:
            continue
        task_id = int(m.group(1))
        description = m.group(2).strip().strip("[]")
        deps_str = (m.group(3) or "").strip()
        dependencies = []
        if deps_str and deps_str.lower() not in ("无", "none", ""):
            for dep in re.split(r"[,，\s]+", deps_str):
                dep = dep.strip()
                if dep.isdigit():
                    dependencies.append(int(dep))
        tasks.append(
            SpecTask(
                task_id=task_id,
                description=description,
                dependencies=dependencies,
            )
        )
    return tasks
