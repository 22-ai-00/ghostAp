"""Discovery and spec generation logic for Spec Engine."""

import json
import logging
import uuid
from typing import Callable, Optional

from ..acp import ACPEvent, ACPEventType
from ..utils.errors import get_error_detail
from .models import SpecProject, SpecWorkItem, SpecWorkItemStatus
from .persistence import persist_generated_spec_file, read_text_file_best_effort
from .utils import extract_json_blob, validate_spec_artifact_dict

logger = logging.getLogger(__name__)


def pick_next_work_item(project: Optional[SpecProject], cycle_num: int) -> Optional[SpecWorkItem]:
    if not project:
        return None
    for wi in project.work_items:
        if wi.status == SpecWorkItemStatus.PENDING and wi.spec_path:
            wi.status = SpecWorkItemStatus.IN_PROGRESS
            wi.used_in_cycle = cycle_num
            return wi
    return None


def should_load_spec_directly(work_item: SpecWorkItem) -> bool:
    path = (work_item.spec_path or "").lower()
    return path.endswith(".json")


def build_input_from_spec_file(original_requirement: str, work_item: SpecWorkItem) -> str:
    spec_text = read_text_file_best_effort(work_item.spec_path)
    return (
        f"## 长程任务目标\n{original_requirement}\n\n"
        f"## 本轮优化关注点（由问题发现机制生成）\n{work_item.question}\n\n"
        f"## 已生成的 Spec 产物（供参考，可修正）\n{spec_text}\n"
    )


def discover_optimization_questions(
    *,
    project: SpecProject,
    session,
    send_prompt_fn: Callable,
    last_review,
    cycle_num: int,
    settings,
    all_satisfied: bool = False,
    backlog_pending: int = 0,
    min_cycles: int = 2,
) -> list[dict]:
    if not session or not project:
        return []

    # === Discovery 门控（防空转）===
    # Gate 1: AC 全满足 → 不再发现新问题
    # BUT: 在前 min_cycles 轮内不应用此门控，因为早期评估可能不可靠
    # （LLM 在同一 session 内评估自己的工作，容易过度乐观）
    gate_on_satisfied = getattr(settings, "spec_discovery_gate_on_satisfied", True)
    if gate_on_satisfied and all_satisfied and cycle_num > min_cycles:
        logger.debug("[Spec] Discovery 门控：所有标准已满足，跳过")
        return []

    # Gate 2: backlog 达上限 → 背压
    max_pending = getattr(settings, "spec_discovery_max_pending", 5)
    if backlog_pending >= max_pending:
        logger.debug(
            "[Spec] Discovery 门控：backlog(%d) >= 上限(%d)，跳过",
            backlog_pending,
            max_pending,
        )
        return []

    # Gate 3: 连续无进展时冷却（每 N 轮触发一次）
    cooldown = getattr(settings, "spec_discovery_cooldown_cycles", 3)
    if cooldown > 1 and project.metrics_history:
        recent_no_progress = 0
        for m in reversed(project.metrics_history):
            if m.new_satisfied == 0:
                recent_no_progress += 1
            else:
                break
        if recent_no_progress > 0 and recent_no_progress % cooldown != 0:
            logger.debug(
                "[Spec] Discovery 冷却中：连续 %d 轮无新进展，每 %d 轮触发一次",
                recent_no_progress,
                cooldown,
            )
            return []

    tracker = project.criteria_tracker
    unsatisfied = tracker.unsatisfied_criteria
    pending_suggestions: list[str] = []
    if last_review:
        for pr in last_review.failed_perspectives:
            for s in pr.suggestions:
                if s:
                    pending_suggestions.append(str(s))
    unsat_text = "\n".join(f"- {c}" for c in unsatisfied[:12]) if unsatisfied else "(无)"
    sugg_text = "\n".join(f"- {s}" for s in pending_suggestions[:12]) if pending_suggestions else "(无)"

    prompt = f"""你是一个长期任务的自我改进系统（Spec-kit 驱动）。

请在本轮 spec-kit 实现后，自动发现与目标相关的"可优化问题"，并提出下一步要解决的关键问题。

## 长程目标
{project.requirement}

## 当前未满足的验收标准（Top）
{unsat_text}

## 上轮审查未通过的建议（Top）
{sugg_text}

## 输出要求（必须严格遵守）
仅输出一个 JSON 数组，放在 ```json fenced code block``` 中，不要输出任何其他文字。

每个元素 schema：
{{
  "id": "Q-...",
  "question": "与目标相关的具体可优化问题（可落到代码/测试/鲁棒性/性能/体验/可维护性）",
  "why": "为什么重要（1-2句）",
  "priority": "P0|P1|P2"
}}

约束：
- 每个 question 必须"可行动"（能转成下一轮 spec-kit 的 Spec 任务单元）
- 优先覆盖未满足验收标准与审查建议
- 数量 1~{int(settings.spec_discovery_max_questions or 5)}
"""

    chunks: list[str] = []

    def on_event(event: ACPEvent):
        if event.event_type == ACPEventType.TEXT_CHUNK and event.text:
            chunks.append(event.text)

    from ..utils.retry import RetryPolicy
    try:
        send_prompt_fn(
            prompt,
            on_event=on_event,
            timeout=120,
            retry_policy=RetryPolicy(max_retries=1, retry_delay=2.0),
        )
        raw = "".join(chunks)
        blob = extract_json_blob(raw)
        data = json.loads(blob) if blob else None
        if isinstance(data, list):
            cleaned: list[dict] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("question", "")).strip()
                if not q:
                    continue
                cleaned.append(
                    {
                        "id": str(item.get("id") or f"Q-{cycle_num}-{len(cleaned) + 1}"),
                        "question": q,
                        "why": str(item.get("why", "")).strip(),
                        "priority": str(item.get("priority", "P1")).strip().upper(),
                    }
                )
            if cleaned:
                return cleaned[: settings.spec_discovery_max_questions]
    except Exception as e:
        logger.debug("[Spec] 问题发现机制失败: %s", get_error_detail(e))

    if settings.spec_discovery_force_nonempty and project:
        fallback_q = None
        if unsatisfied:
            fallback_q = f"如何满足验收标准：{unsatisfied[0]}？"
        elif pending_suggestions:
            fallback_q = f"如何落实改进建议：{pending_suggestions[0]}？"
        else:
            fallback_q = "当前实现还有哪些可测试性/可维护性/鲁棒性方面的改进空间？"
        return [
            {
                "id": f"Q-{cycle_num}-1",
                "question": fallback_q,
                "why": "兜底：保证长程任务每轮都有明确的下一步优化方向",
                "priority": "P1",
            }
        ]

    return []


def generate_specs_from_discovery(
    *,
    project: SpecProject,
    session,
    send_prompt_fn: Callable,
    root_path: str,
    settings,
    cycle_num: int,
    discovery: list[dict],
) -> list[SpecWorkItem]:
    if not project:
        return []
    if not discovery:
        return []

    max_specs = settings.spec_generated_specs_per_cycle
    selected = discovery[:max_specs]

    if not session:
        return []

    prompt = f"""你是一个 spec-kit 规格生成器。

请把下面这些"可优化问题"拆解为符合 spec-kit 的 Spec 任务单元，并为每个问题生成一个 Spec JSON 产物。

## 长程目标
{project.requirement}

## 可优化问题列表
{json.dumps(selected, ensure_ascii=False, indent=2)}

## 输出要求（必须严格遵守）
仅输出一个 JSON 数组，放在 ```json fenced code block``` 中，不要输出任何其他文字。

数组元素 schema：
{{
  "id": "Q-...",
  "spec": {{
    "goals": ["..."],
    "functional_spec": ["..."],
    "non_functional_requirements": ["..."],
    "acceptance_criteria": ["可验证条件..."],
    "out_of_scope": ["..."],
    "risks": ["..."],
    "clarification_questions": ["..."],
    "decisions": ["..."],
    "required_experts": [{{"role":"任务真正需要的专家", "purpose":"该角色为何必要", "focus":["具体关注点"], "checks":["必须验证的事项"]}}],
    "version": "1.0"
  }}
}}

每个 spec 的 required_experts 必须有 3–7 个任务推断出的独立角色；名称归一化后不得重复，且每项 role、purpose、focus、checks 均非空。写作问题必须涵盖创意/选题、目标读者、编辑、审稿/事实或论证核查四种视角。
"""

    chunks: list[str] = []

    def on_event(event: ACPEvent):
        if event.event_type == ACPEventType.TEXT_CHUNK and event.text:
            chunks.append(event.text)

    items: list[SpecWorkItem] = []
    from ..utils.retry import RetryPolicy
    try:
        send_prompt_fn(
            prompt,
            on_event=on_event,
            timeout=180,
            retry_policy=RetryPolicy(max_retries=1, retry_delay=2.0),
        )
        raw = "".join(chunks)
        blob = extract_json_blob(raw)
        data = json.loads(blob) if blob else None
        if not isinstance(data, list):
            data = []

        id_to_question = {str(d.get("id")): str(d.get("question", "")).strip() for d in selected}

        for entry in data:
            if not isinstance(entry, dict):
                continue
            qid = str(entry.get("id") or "").strip()
            spec = entry.get("spec")
            if not qid or not isinstance(spec, dict):
                continue
            errors = validate_spec_artifact_dict(spec)
            if errors:
                logger.debug("[Spec] 生成的 spec 产物不合规(qid=%s): %s", qid, errors[:3])
            question = id_to_question.get(qid) or qid
            spec_text = json.dumps(spec, ensure_ascii=False, indent=2)
            spec_path = persist_generated_spec_file(root_path, settings, project, cycle_num, qid, spec_text)
            items.append(
                SpecWorkItem(
                    item_id=qid,
                    question=question,
                    created_in_cycle=cycle_num,
                    spec_path=spec_path,
                    status=SpecWorkItemStatus.PENDING,
                )
            )

    except Exception as e:
        logger.debug("[Spec] spec 生成失败: %s", get_error_detail(e))

    if not items and settings.spec_discovery_force_nonempty:
        for d in selected[:max_specs]:
            qid = str(d.get("id") or f"Q-{cycle_num}-{uuid.uuid4().hex[:4]}")
            question = str(d.get("question") or "").strip() or qid
            minimal = {
                "goals": [f"解决问题：{question}"],
                "functional_spec": ["完成满足问题要求的必要交付"],
                "non_functional_requirements": ["不引入回归，保持可验证性"],
                "acceptance_criteria": [f"问题被解决：{question}"],
                "out_of_scope": [],
                "risks": [],
                "clarification_questions": [],
                "decisions": ["假设：允许基于现有实现做增量改进"],
                "version": "1.0",
            }
            spec_path = persist_generated_spec_file(
                root_path, settings, project, cycle_num, qid, json.dumps(minimal, ensure_ascii=False, indent=2)
            )
            items.append(
                SpecWorkItem(
                    item_id=qid,
                    question=question,
                    created_in_cycle=cycle_num,
                    spec_path=spec_path,
                    status=SpecWorkItemStatus.PENDING,
                )
            )

    return items
