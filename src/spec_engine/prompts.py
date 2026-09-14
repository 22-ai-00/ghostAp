"""Prompt construction functions for the Spec Engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from ..engine_base import ReviewPerspective, ReviewResult
from ..grill_me import (
    SPEC_BUILD_GRILL_ME_PROTOCOL,
    SPEC_PHASE_GRILL_ME_PROTOCOL,
    SPEC_REVIEW_GRILL_ME_PROTOCOL,
    SPEC_TASK_GRILL_ME_PROTOCOL,
)

if TYPE_CHECKING:
    from .models import PlanArtifact, SpecArtifact, SpecProject, SpecTask


def format_criteria_status(project: Optional[SpecProject]) -> str:
    if not project or not project.criteria_tracker.criteria:
        return ""
    tracker = project.criteria_tracker
    lines = ["\n## 验收标准进度"]
    for i, c in enumerate(tracker.criteria):
        if tracker.satisfied.get(i, False):
            lines.append(f"- [x] {c} ✅ (已满足)")
        else:
            lines.append(f"- [ ] {c}")
    return "\n".join(lines) + "\n"


def build_spec_prompt(requirement: str, root_path: str, guidance: str, criteria_status: str) -> str:
    return f"""你是跨领域的需求与交付架构师。请使用 spec-kit 风格产出"规格（Spec）"；任务可以是写作、研究、设计、运营、规划或软件工作，绝不可预设为编码任务。

目标：只描述 **做什么/为什么/范围与约束**，不讨论具体怎么实现。

## 需求
{requirement}

## 工作目录
{root_path}
{guidance}{criteria_status}
{SPEC_PHASE_GRILL_ME_PROTOCOL}
## 输出要求（必须严格遵守）
仅输出一个 JSON 对象，放在 ```json fenced code block``` 中，不要输出任何其他文字。

Schema（字段必须存在；required_experts 为对象数组，其余数组元素为字符串）：
{{
  "goals": ["..."],
  "functional_spec": ["..."],
  "non_functional_requirements": ["..."],
  "acceptance_criteria": ["可验证条件..."],
  "out_of_scope": ["明确不做什么..."],
  "risks": ["风险/约束..."],
  "clarification_questions": ["已识别的模糊点（仅记录，不等待用户回答）..."],
  "decisions": ["已确认/可接受的假设（必须显式标注为假设）..."],
  "required_experts": [{{"role":"任务真正需要的专家角色", "purpose":"该角色为何对本任务必要", "focus":["具体关注点"], "checks":["该角色必须验证的事项"]}}],
  "version": "1.0"
}}

约束：
- acceptance_criteria 必须可被判定 PASS/FAIL
- 遇到信息不足时，不要停下等待用户——基于项目上下文和行业最佳实践自主选择最优方案
- 将模糊点记录在 clarification_questions 中（仅供参考），将你的决策记录在 decisions 中
- 如果用户引导中提供了相关信息，优先使用用户的指示
- 先从目标、受众、交付物、风险和验收标准推断完成与深度审查本任务所需的 3–7 个不同专家；角色必须来自任务语境，不能套用固定的软件、产品或设计角色模板
- 写作任务必须在 required_experts 中包含创意/选题、目标读者、编辑、审稿/事实或论证核查四种独立视角，并写明各自理由
"""


def build_plan_prompt(spec: str, root_path: str, spec_artifact: Optional["SpecArtifact"] = None) -> str:
    if spec_artifact and spec_artifact.goals:
        spec_section = json.dumps(spec_artifact.to_dict(), ensure_ascii=False, indent=2)
    else:
        spec_section = spec

    return f"""你是跨领域交付规划师。基于下述 Spec（规格），产出 Plan（规划），强调可执行、可验证；按任务本身选择合适的方法、材料、工具和交付物，不要假定这是软件工程。

## Spec 输入
{spec_section}

## 工作目录
{root_path}

{SPEC_PHASE_GRILL_ME_PROTOCOL}
## 输出要求（必须严格遵守）
仅输出一个 JSON 对象，放在 ```json fenced code block``` 中，不要输出任何其他文字。

Schema（字段必须存在；数组元素为字符串）：
{{
  "architecture": "总体方法、结构或关键决策（文本）",
  "tech_stack": ["适用的工具、材料、渠道、语言、框架或资料..."],
  "steps": ["按优先级的一句话步骤..."],
  "file_changes": ["将创建/修改的文件、交付物或工作对象..."],
  "test_plan": ["验证、审阅、试读、试用、测量或其他验收方式..."],
  "risks": ["风险与应对..."],
  "version": "1.0"
}}

规划约束：
- steps 中请优先识别可并行的独立工作包；如果当前工具支持 subagent / 子任务委托，应把无共享交付物、无依赖顺序冲突的工作设计为可委托执行
- 对可能冲突的工作（同一交付物、同一事实来源、同一受众承诺或同一配置入口）必须在 steps 或 risks 中标注串行顺序与合并风险
"""


def build_task_prompt(plan: str, plan_artifact: Optional["PlanArtifact"] = None) -> str:
    if plan_artifact and plan_artifact.steps:
        plan_section = json.dumps(plan_artifact.to_dict(), ensure_ascii=False, indent=2)
    else:
        plan_section = plan

    return f"""将以下方案分解为可执行的具体任务。任务可以是任何领域的研究、创作、设计、运营或实现工作；保持原方案的领域语境。

## 实现方案
{plan_section}

{SPEC_TASK_GRILL_ME_PROTOCOL}
## 输出要求
请输出结构化的任务列表，每个任务包含：
- 任务编号（1, 2, 3, ...）
- 任务描述（一句话）
- 依赖的任务编号（如果有）

格式（严格遵循）：
1. [任务描述] (依赖: 无)
2. [任务描述] (依赖: 1)
3. [任务描述] (依赖: 1, 2)
...

要求：
- 每个任务应有独立的验收或审阅方式
- 任务粒度适中，不要过大或过小
- 标注依赖关系以确定执行顺序；没有依赖且不争用同一交付物、事实来源或关键决策的任务应被拆成可并行、可委托给 subagent 的独立任务
- 如果任务存在潜在冲突，请用依赖关系把它们串起来，不要把冲突任务伪装成并行任务
"""


def build_build_prompt(tasks: list[SpecTask], plan: str, root_path: str, guidance: str, plan_artifact: Optional["PlanArtifact"] = None) -> str:
    task_list = "\n".join(f"{t.task_id}. {t.description}" for t in tasks)
    if plan_artifact and plan_artifact.steps:
        plan_section = json.dumps(plan_artifact.to_dict(), ensure_ascii=False, indent=2)
    else:
        plan_section = plan

    return f"""按以下任务列表逐步完成交付；保持任务原有领域，不要把非编码任务改写成代码实现。

## 实现方案
{plan_section}

## 任务列表
{task_list}

## 工作目录
{root_path}
{guidance}
{SPEC_BUILD_GRILL_ME_PROTOCOL}
## 要求
1. 严格按照依赖关系推进；不要让有冲突的任务并发修改同一交付物、同一事实来源、同一受众承诺或同一迁移/配置入口
2. 对依赖已满足且改动范围互不冲突的任务，优先使用当前工具支持的 subagent / 子任务委托并行执行
3. 如果无法使用 subagent，也要按可并行工作包组织执行，避免把互不相关的任务强制串行化
4. 每个任务完成后进行自检
5. 确保交付质量：准确、适合目标受众、遵守领域约束，并对适用风险采取措施
6. 完成所有任务后输出总结，说明哪些任务并行/委托执行、哪些任务因冲突或依赖而串行执行，以及每项验收证据
"""


def build_review_prompt(requirement: str) -> str:
    perspective_sections = []
    for p in ReviewPerspective:
        if p == ReviewPerspective.PRODUCT:
            perspective_sections.append(
                "- **PRODUCT**: Apple 风格产品审查（高审美/高标准/完美主义）。关注：信息架构与心智模型、关键路径是否一气呵成、默认行为是否聪明、边界与异常是否体面、文案是否克制清晰、细节一致性与打磨程度。"
            )
        else:
            perspective_sections.append(f"- **{p.value.upper()}**: {p.review_focus}")
    perspectives_desc = "\n".join(perspective_sections)

    return f"""这是兼容旧调用的通用交付审查提示。请从任务实际需要的专业、受众、验收、风险和可用性视角审查当前交付质量，并给出结构化的审查结果；不要假设任务是软件实现。

## 项目目标
{requirement}

## 审查视角
{perspectives_desc}

## PRODUCT 视角加严要求（Apple 风格）
- 以"少即是多"的审美判断：删繁就简，不为功能堆砌找理由。
- 以默认体验为王：默认路径必须顺滑、可预期、可解释；拒绝把复杂度转嫁给用户。
- 以细节一致性为底线：命名/状态/交互/错误提示/边界行为必须统一。
- 以体面为标准：失败与异常也要有尊严（清晰提示、可恢复、不给用户添堵）。
- 建议要具体可落地：每条建议最好能对应到 1 个明确改动点（文案/交互/流程/边界/信息层级）。

{SPEC_REVIEW_GRILL_ME_PROTOCOL}

<output_format>
严格按照以下格式输出每个视角的审查结果（每个视角占一个区块）。
不要使用 markdown 表格、JSON、编号列表等任何其他格式。
必须使用 [TAG] 作为区块分隔符。

[ARCHITECT]
PASS 或 FAIL
- 改进建议1（如果FAIL）
- 改进建议2（如果FAIL）

[PRODUCT]
PASS 或 FAIL
- 改进建议1（如果FAIL）

[USER]
PASS 或 FAIL
- 改进建议1（如果FAIL）

[TESTER]
PASS 或 FAIL
- 改进建议1（如果FAIL）

[DESIGNER]
PASS 或 FAIL
- 改进建议1（如果FAIL）
- 改进建议2（如果FAIL）
- (请重点关注: UI视觉、交互体验、移动端适配)
</output_format>

<example>
[ARCHITECT]
PASS

[PRODUCT]
FAIL
- 缺少错误提示文案
- 搜索结果无分页

[USER]
PASS

[TESTER]
FAIL
- 缺少边界条件测试

[DESIGNER]
FAIL
- 按钮间距过小，容易误触
- 错误提示颜色对比度不足
</example>

## 审查标准
- PASS: 该视角认为当前实现质量良好，无需改进
- FAIL: 该视角发现可改进之处，请列出具体建议
- 建议应具体、可操作，而非泛泛而谈
- 如果某视角为 PASS，不需要列出建议
"""


def build_single_perspective_review_prompt(
    perspective: ReviewPerspective,
    *,
    requirement: str,
    diff_patch: str = "",
    touched_files: Optional[list[str]] = None,
    spec_output: str = "",
    plan_output: str = "",
    build_output: str = "",
    max_diff_bytes: int = 40_000,
) -> str:
    """Single-perspective review prompt. Small context, single [TAG] output.

    Designed for PerspectiveWorker — each worker sends exactly this prompt to
    its own (possibly ephemeral) session. Avoids the 5-in-1 context bloat that
    caused Invalid params / timeout cascades.
    """
    tag = perspective.name  # ARCHITECT / PRODUCT / USER / TESTER / DESIGNER
    if perspective == ReviewPerspective.PRODUCT:
        focus_block = (
            "Apple 风格产品审查（高审美/高标准/完美主义）。关注：信息架构与心智模型、"
            "关键路径是否一气呵成、默认行为是否聪明、边界与异常是否体面、文案是否克制"
            "清晰、细节一致性与打磨程度。"
        )
    else:
        focus_block = perspective.review_focus

    diff_section = ""
    if diff_patch:
        patch = diff_patch
        if len(patch) > max_diff_bytes:
            patch = patch[:max_diff_bytes] + f"\n...[truncated {len(diff_patch) - max_diff_bytes} bytes]"
        diff_section = f"\n## 代码变更 (git diff HEAD)\n```diff\n{patch}\n```\n"

    files_section = ""
    if touched_files:
        files_section = "\n## 涉及文件\n" + "\n".join(f"- {f}" for f in touched_files[:50]) + "\n"

    spec_section = f"\n## Spec 摘要\n{spec_output}\n" if spec_output else ""
    plan_section = f"\n## Plan 摘要\n{plan_output}\n" if plan_output else ""
    build_section = f"\n## Build 摘要\n{build_output}\n" if build_output else ""

    return f"""你是 **{perspective.display_name}** 视角的评审员。只从这一个视角给出判断，不要覆盖其它视角。

## 项目目标
{requirement}

## 视角关注点
{focus_block}
{spec_section}{plan_section}{build_section}{files_section}{diff_section}
<output_format>
严格按以下格式输出，不要输出其他任何内容：

[{tag}]
PASS 或 FAIL
- 改进建议1（如果 FAIL）
- 改进建议2（如果 FAIL）
</output_format>

## 标准
- PASS: 本视角认为当前实现质量良好，无需改进
- FAIL: 发现可改进之处，列出具体、可落地的建议（每条对应一个明确改动点）
"""


def build_refinement_input(
    original_requirement: str,
    last_review: Optional[ReviewResult],
    project: Optional[SpecProject],
) -> str:
    lines = [f"## 原始需求\n{original_requirement}\n"]

    if last_review:
        failed = last_review.failed_perspectives
        if failed:
            lines.append("## 上轮审查改进建议\n以下建议需要在本轮 Spec 循环中解决：\n")
            for pr in failed:
                label = pr.role_display_name or pr.perspective.display_name
                lines.append(f"{pr.perspective.emoji} **{label}**:")
                for s in pr.suggestions:
                    lines.append(f"  - {s}")
                lines.append("")

    if project:
        tracker = project.criteria_tracker
        unsatisfied = tracker.unsatisfied_criteria
        if unsatisfied:
            lines.append("## 未满足的验收标准\n")
            for c in unsatisfied:
                lines.append(f"- [ ] {c}")
            lines.append("")

    return "\n".join(lines)
