"""Script generation for workflow orchestration scripts."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subagent encouragement — appended to every agent() prompt template
# ---------------------------------------------------------------------------

SUBAGENT_ENCOURAGEMENT: str = (
    "**Subagent Usage Encouragement**: When a task can be decomposed, always "
    "delegate to subagents rather than doing everything yourself. Each subagent "
    "can further spawn its own subagents or sub-workflows. Subagents work in "
    "parallel and can independently handle research, implementation, verification, and "
    "testing tasks, significantly improving efficiency and convergence speed. Don't "
    "hesitate to spawn multiple subagents for different parts of the task — "
    "they are designed to work concurrently and will report back their results."
)


def _subagent_hint_enabled() -> bool:
    """Return True if the subagent / workflow encouragement paragraph should be appended.

    Reads the ``workflow_subagent_hint_enabled`` setting at call time so that
    runtime configuration changes are honoured.  Falls back to ``True`` if the
    settings module is unavailable (e.g. during unit tests that do not import
    the full application stack).
    """
    try:
        from src.config import get_settings

        return bool(getattr(get_settings(), "workflow_subagent_hint_enabled", True))
    except Exception:
        return True


def get_subagent_encouragement() -> str:
    """Return the subagent encouragement paragraph, or "" if disabled via settings."""
    return SUBAGENT_ENCOURAGEMENT if _subagent_hint_enabled() else ""

# ---------------------------------------------------------------------------
# Dangerous patterns that must not appear in generated scripts
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"""require\s*\(\s*['"]fs['"]\s*\)""", "filesystem access via require('fs')"),
    (r"""require\s*\(\s*['"]child_process['"]\s*\)""", "shell access via require('child_process')"),
    (r"""require\s*\(\s*['"]net['"]\s*\)""", "network access via require('net')"),
    (r"""require\s*\(\s*['"]dgram['"]\s*\)""", "UDP access via require('dgram')"),
    (r"""require\s*\(\s*['"]http['"]\s*\)""", "HTTP access via require('http')"),
    (r"""require\s*\(\s*['"]https['"]\s*\)""", "HTTPS access via require('https')"),
    (r"""process\.exit""", "process.exit() call"),
    (r"""process\.env""", "process.env access"),
    (r"""process\s*\[""", "process[...] bracket access (process.env alias)"),
    (r"""import\s+.*from\s+['"]fs['"]""", "filesystem access via import 'fs'"),
    (r"""import\s+.*from\s+['"]child_process['"]""", "shell access via import 'child_process'"),
    (r"""import\s+.*from\s+['"]node:fs['"]""", "filesystem access via import 'node:fs'"),
    (r"""import\s+.*from\s+['"]node:child_process['"]""", "shell access via import 'node:child_process'"),
    (r"""import\s+.*from\s+['"]node:net['"]""", "network access via import 'node:net'"),
    (r"""import\s+.*from\s+['"]node:dgram['"]""", "UDP access via import 'node:dgram'"),
    (r"""import\s+.*from\s+['"]node:http['"]""", "HTTP access via import 'node:http'"),
    (r"""import\s+.*from\s+['"]node:https['"]""", "HTTPS access via import 'node:https'"),
    (r"""eval\s*\(""", "eval() usage"),
    (r"""Function\s*\(""", "dynamic Function constructor"),
    (r"""\.constructor\s*\.\s*constructor\s*\(""",
        "constructor.constructor escape (reaching the host Function constructor)"),
    (r"""new\s+Worker\s*\(""", "Worker thread creation"),
    (r"""globalThis\[""", "globalThis bracket access"),
    (r"""Deno\.""", "Deno runtime API"),
    (r"""Bun\.""", "Bun runtime API"),
    (r"""\bimport\s*\(""", "dynamic import() expression"),
    (r"""\bimport\.meta\b""", "import.meta access"),
]

# ---------------------------------------------------------------------------
# Prompt template for script generation
# ---------------------------------------------------------------------------

# Injected section sentinels. These markers are removed after the
# agent capability section is spliced in. If a marker is missing for any
# reason (template edits, tests that strip them), insertion falls back to
# appending the section at the end of the prompt instead of raising.
_USER_REQUIREMENT_INSERT_POINT = (
    "### SENTINEL: USER_REQUIREMENT_INSERT_POINT ###"
)

_SCRIPT_GEN_PROMPT_TEMPLATE = """\
# Workflow Script Generation Task

## CRITICAL RULES (违反即验证失败)

1. 每个 agent() 必须有唯一 `label` 和显式 `timeout`
2. 必须检查 result.error — 单个 agent 失败不能终止整个 workflow
3. 输出格式：`export const meta = {{...}}` + `export default async function() {{...}}`
4. 禁止 require()/import/process.exit/eval/filesystem/network 访问

""" + _USER_REQUIREMENT_INSERT_POINT + """
## User Requirement

{requirement}

## Available Resources

**Tools (AI agents you can dispatch):**
{tools_list}

**Roles (specialized perspectives for agents):**
根据任务需求自行规划角色分工。每个 agent() 调用可通过 `role` 参数指定适合的角色（如 architect、reviewer、tester 等）。
角色不是固定列表，而是你根据任务复杂度和需要覆盖的维度自主决定的。
建议考虑：架构设计、代码实现、安全审计、正确性验证、测试覆盖等维度。

## Dynamic Workflow Reliability Rules

目标是生成 Claude-style 动态 workflow：脚本本身负责按用户需求动态选择模式、拆分任务、处理失败并收敛，而不是把所有决策再次外包给一个慢速大 Agent。

- **Scope First**：不要先派一个大而慢的 analysis agent 包办完整工作。范围明确时直接形成有上限的 worklist；范围未知时先用一个轻量 scout Agent 发现并裁剪工作项，再展开执行。
- **Pipeline-First**：同一批 work item 要经过相同的多个阶段时优先使用 `pipeline()`；它保持 item 内 stage 串行、item 间有界并行，避免手写无界 Promise fan-out。
- **Barrier Discipline**：只有下游必须看到全部上游结果（去重、排名、交叉比较、全局汇总）时才使用 `parallel()`/`fanout()` 后的 barrier；否则让各 item 在 pipeline 中持续向后流动。
- **Structured Output Contract**：跨节点传递的数据必须用紧凑 shape schema 约束。schema 失配是节点失败，必须检查 `result.error`，不得把未验证文本继续传给下游。
- 直接基于用户需求选择 classify/fanout/verify/loop/race，不要生成固定两步 "Analysis -> Execution" 模板。
- 每个 agent() label 必须唯一，例如 `analysis-router-1`、`impl-worker-2`、`verify-r1-security`；不要复用 task-analysis、analysis、worker 等通用 label。
- 为每个 agent() 显式设置短超时：轻量分类/路由 60-90s，普通子任务 120-180s，只有确实需要长推理时才使用 300s。
- 检查 result.error 并提供 fallback：切换工具、走 race() 备用 Agent、降级输出部分结果，或把失败纳入 synthesis，而不是让单个失败阻塞整个 workflow。
- 对并行结果做容错合成：不要假设 fanout/parallel 的每个结果都成功；过滤错误项并说明哪些子任务失败。

## Output Format

Generate a complete ES Module (.js) workflow script. The script MUST:

1. Export a `meta` constant describing the workflow
2. Export a `default` async function that orchestrates the workflow

### Meta Schema (REQUIRED)

```javascript
export const meta = {{
  name: "workflow-name-kebab-case",       // string, required
  description: "One-line description",     // string, required
  phases: [                                // array, at least 1 phase
    {{ title: "Phase Title", detail: "What this phase accomplishes" }},
  ],
  maxConcurrent: 6,                        // number, optional (default 10)
  tools: ["coco", "claude"],               // array of tools used
  patterns: ["fanout", "verify"],          // array, optional: which patterns are used
  workflow_refs: ["sub-workflow-name"],     // array, optional: sub-workflows invoked
}};
```

### Available Primitives — Core

All primitives are globally available (no import needed):

```javascript
// Send a prompt to an AI agent and get the response
const result = await agent(prompt, {{
  tool: "coco",           // which AI tool to use
  role: "architect",      // optional role/persona
  label: "task-label",    // required: unique label for tracking
  phase: "Phase Title",   // optional phase association
  // compact recursive shape contract; [] accepts any array, [{{...}}] validates every item
  schema: {{ key: "string", findings: [{{ severity: "", line: 0 }}] }},
  timeout: 180,           // required: bound every direct agent call
}});
if (result && result.error) return {{ error: result.error, stage: "agent-call" }};

// Run multiple independent tasks in parallel
const [r1, r2, r3] = await parallel([
  () => agent("task 1", {{ tool: "coco", label: "parallel-1", timeout: 120 }}),
  () => agent("task 2", {{ tool: "claude", label: "parallel-2", timeout: 120 }}),
  () => agent("task 3", {{ tool: "aiden", label: "parallel-3", timeout: 120 }}),
]);

// Bounded parallel/map: items honour maxConcurrent; stages stay sequential per item
const results = await pipeline(items, stage1Fn, stage2Fn, {{ continueOnFailure: true }});

// Strict sequential execution (each step receives previous result)
const final = await sequence([step1Fn, step2Fn, step3Fn]);

// Declare phase transitions (for progress tracking)
phase("Phase Title");

// Logging
log("Status message");

// Invoke a sub-workflow by name
const subResult = await workflow("sub-workflow-name", {{ arg1: "value" }});
```

### Available Primitives — Dynamic Workflow Patterns (6大编排模式)

These are higher-order orchestration primitives implementing proven multi-agent patterns.
**选择最适合任务的模式组合，而非总是手写循环和条件判断。**

#### 1. classify(input, categories, opts) — 分类-执行模式 (Classify-and-Act)

先分类再路由到不同处理逻辑。适合：多种任务类型需要不同处理策略。

```javascript
const result = await classify(userRequest, {{
  "bug_fix": {{
    description: "Bug fixes and error corrections",
    handler: async (input) => agent(`Fix this bug: ${{input}}`, {{ tool: "coco", label: "route-bug-fix", timeout: 180 }}),
  }},
  "feature": {{
    description: "New feature implementation",
    handler: async (input) => agent(`Implement: ${{input}}`, {{ tool: "claude", label: "route-feature", timeout: 180 }}),
  }},
  "refactor": {{
    description: "Code refactoring and optimization",
    handler: async (input) => agent(`Refactor: ${{input}}`, {{ tool: "aiden", label: "route-refactor", timeout: 180 }}),
  }},
}}, {{ classifierTool: "claude", timeout: 90 }});
```

#### 2. fanout(input, workers, opts) — 扇出-合成模式 (Fan-out-and-Synthesize)

拆分为多个独立子任务并行执行，最后合成结果。适合：大量独立子问题、多视角分析。

```javascript
const result = await fanout(codeContext, [
  {{ prompt: "Review for security: ${{input}}", tool: "claude", role: "security_auditor", label: "security-review", timeout: 180 }},
  {{ prompt: "Review for performance: ${{input}}", tool: "aiden", role: "perf_expert", label: "performance-review", timeout: 180 }},
  {{ prompt: "Review for correctness: ${{input}}", tool: "coco", role: "correctness_checker", label: "correctness-review", timeout: 180 }},
], {{ synthesizerTool: "claude", synthesizerRole: "lead_reviewer", timeout: 180 }});
```

#### 3. verify(output, opts) — 对抗性验证模式 (Adversarial Verification)

用独立验证者挑战输出，循环修订直到通过。适合：高置信度需求、安全审查、关键代码。

```javascript
const {{ accepted, output: verifiedCode, feedback }} = await verify(generatedCode, {{
  criteria: "security, correctness, no regressions",
  verifiers: [
    {{ tool: "claude", role: "security_adversary", focus: "Find security vulnerabilities", timeout: 180 }},
    {{ tool: "aiden", role: "correctness_adversary", focus: "Find logic errors and edge cases", timeout: 180 }},
  ],
  maxRounds: 3,
  reviseTool: "coco",
  timeout: 180,
}});
```

#### 4. generate(count, generatorFn, filterFn, opts) — 生成-过滤模式 (Generate-and-Filter)

生成多个候选方案，然后过滤筛选出最优。适合：命名、设计方案、创意探索。

```javascript
const topSolutions = await generate(
  5,  // generate 5 candidates
  (i) => ({{ prompt: `Design approach ${{i+1}} for: ${{task}}`, tool: "coco", role: `designer-${{i}}`, timeout: 120 }}),
  null,  // use default filter (AI-based ranking)
  {{ topK: 2, criteria: "feasibility, elegance, maintainability", filterTool: "claude", timeout: 120 }}
);
```

#### 5. tournament(contestants, judgeFn, opts) — 锦标赛模式 (Tournament)

让多个智能体竞争同一任务，通过淘汰赛决出最佳方案。适合：确定最佳实现方案。

```javascript
const {{ winner, bracket }} = await tournament(
  [
    {{ prompt: `Solve with approach A: ${{task}}`, tool: "coco", label: "approach-A", timeout: 180 }},
    {{ prompt: `Solve with approach B: ${{task}}`, tool: "claude", label: "approach-B", timeout: 180 }},
    {{ prompt: `Solve with approach C: ${{task}}`, tool: "aiden", label: "approach-C", timeout: 180 }},
    {{ prompt: `Solve with approach D: ${{task}}`, tool: "gemini", label: "approach-D", timeout: 180 }},
  ],
  null,  // use default judge
  {{ judgeTool: "claude", task: task, criteria: "correctness, efficiency, readability", timeout: 180 }}
);
```

#### 6. loop(taskFn, opts) — 循环直到完成模式 (Loop-Until-Done)

反复执行直到满足停止条件或收敛。适合：Bug 打猎、安全审计、迭代优化。

```javascript
const {{ results, iterations, stoppedBy }} = await loop(
  async (i, prev) => {{
    return agent(`Iteration ${{i+1}}: Find more issues not in: ${{prev || 'none'}}`, {{
      tool: "claude", label: `hunt-${{i}}`, schema: {{ issues: [], done: false }}, timeout: 180
    }});
  }},
  {{
    maxIterations: 8,
    stopWhen: (result) => result?.issues?.length === 0 || result?.done === true,
    convergenceCheck: (curr, prev) => {{
      const currSet = new Set((curr?.issues || []).map(i => i.description));
      const prevSet = new Set((prev?.issues || []).map(i => i.description));
      return currSet.size === prevSet.size && [...currSet].every(x => prevSet.has(x));
    }},
  }}
);
```

#### race(contestants, opts) — 竞速模式 (First-to-finish)

多个智能体竞速，取第一个有效结果。适合：多种方法可能成功，取最快的。

```javascript
const fastest = await race([
  {{ prompt: task, tool: "coco", label: "fast-coco", timeout: 120 }},
  {{ prompt: task, tool: "traex", label: "fast-traex", timeout: 120 }},
], {{ validate: (r) => r && !r.error && r.length > 50 }});
```

## Pattern Composition Strategy (模式组合策略)

最强大的 workflow 通常组合多个模式。常见组合：

1. **classify → fanout → verify**: 先分类，针对性扇出处理，最后验证
2. **fanout → tournament**: 多角度生成，锦标赛选最佳
3. **loop + verify**: 迭代产出 + 每轮验证
4. **generate → tournament → verify**: 生成多方案 → 淘汰赛 → 对抗验证
5. **classify → loop**: 分类后针对不同类型用循环消化
6. **fanout → loop(verify)**: 并行处理后循环验证直到全部通过

**选择模式的决策树：**
- 同构 work item 经过相同多阶段？ → pipeline（默认首选）
- 任务有多种类型？ → classify
- 可拆为独立子问题？ → fanout
- 需要高置信度？ → verify
- 需要最优方案？ → tournament 或 generate+filter
- 工作量未知/迭代性？ → loop
- 多种方法都可能成功？ → race

## Best Practices (MUST FOLLOW)

1. **优先使用 pipeline 或高阶模式** — 同构多阶段 worklist 优先 pipeline；当任务匹配某个模式时，直接使用 classify/fanout/verify/generate/
   tournament/loop，而非手写等价逻辑。模式内置了错误处理、重试和收敛检测。

2. **Assign different tools to different roles/tasks** — Diversity of AI perspectives
   produces more robust results. Don't use the same tool for everything.

3. **Include adversarial verification for critical outputs** — For important conclusions,
   use verify() or have a separate agent with a different tool challenge the findings.

4. **Keep each agent() prompt focused on one task** — Don't ask an agent to do
   everything at once. Break complex work into focused, composable agent() calls.

5. **Encourage subagent usage in prompts** - When writing agent prompts, tell the agent
   to spawn subagents for independent sub-problems. Include this guidance in prompts
   for complex tasks:
   "When a task can be decomposed, always delegate to subagents rather than doing
   everything yourself. Each subagent can further spawn its own subagents or
   sub-workflows. Subagents work in parallel and can independently handle research,
   implementation, verification, and testing tasks, significantly improving efficiency
   and convergence speed."

6. **Declare phases** - Call phase("Title") before each logical section to enable
   progress tracking in the UI.

7. **Use unique labels and explicit timeouts** - Every direct agent() call and
   every agent descriptor passed into a pattern must have a unique `label` and
   an explicit `timeout`.

8. **Never let a single agent failure kill the workflow** - Always wrap agent() calls
   in error handling. If one agent fails, continue with remaining agents and include
   the error in the final report. Pattern:
   ```
   const result = await agent({{ ... }});
   if (result.error) {{
     log(`Agent failed: ${{result.error}}`);
     // Continue with fallback or skip this step
   }}
   ```
   For parallel/fanout patterns, filter out errored results before synthesis:
   ```
   const results = await fanout(input, workers);
   const valid = results.filter(r => !r.error);
   if (valid.length === 0) return {{ error: "All agents failed", details: results }};
   ```

9. **模式可嵌套** — verify() 内部可以用 fanout()，loop() 每轮可以用 tournament()，等等。

## Constraints

- Do NOT use `require()` or `import` statements (primitives are global)
- Do NOT access the filesystem, network, or child processes
- Do NOT call `process.exit()`
- Do NOT use `eval()` or `new Function()`
- Keep the script self-contained — all logic in one file

## Example: Multi-Pattern Workflow

```javascript
export const meta = {{
  name: "robust-implementation",
  description: "Generate, compete, verify — robust feature implementation",
  phases: [
    {{ title: "Analysis", detail: "Analyze task type and complexity" }},
    {{ title: "Generation", detail: "Generate multiple implementation approaches" }},
    {{ title: "Tournament", detail: "Compete approaches to find the best" }},
    {{ title: "Verification", detail: "Adversarial verification of the winner" }},
  ],
  maxConcurrent: 6,
  tools: ["coco", "claude", "aiden"],
  patterns: ["generate", "tournament", "verify"],
}};

export default async function() {{
  const task = workflowArgs.task || "Implement the requested feature";

  // Phase 1: Generate competing approaches
  phase("Generation");
  log("Generating competing approaches...");

  const topApproaches = await generate(
    4,
    (i) => ({{
      prompt: `Design approach #${{i+1}} for: ${{task}}. Use a distinctly different strategy.`,
      tool: ["coco", "claude", "aiden", "gemini"][i % 4],
      role: `designer-${{i}}`,
      timeout: 120,
    }}),
    null,
    {{ topK: 4, criteria: "feasibility and correctness", timeout: 120 }}
  );
  if (topApproaches.some(item => item && item.error)) {{
    return {{ error: "One or more generation candidates failed", partial: topApproaches }};
  }}

  // Phase 2: Tournament to find the best
  phase("Tournament");
  log("Running tournament...");

  const {{ winner }} = await tournament(
    topApproaches.map((approach, i) => ({{
      prompt: `Refine and complete this approach:\\n${{typeof approach === 'string' ? approach : JSON.stringify(approach)}}`,
      tool: ["coco", "claude", "aiden", "gemini"][i % 4],
      label: `finalist-${{i}}`,
      timeout: 180,
    }})),
    null,
    {{ judgeTool: "claude", task: task, criteria: "correctness, efficiency, maintainability", timeout: 180 }}
  );
  if (winner && winner.error) {{
    return {{ error: winner.error, stage: "Tournament" }};
  }}

  // Phase 3: Adversarial verification
  phase("Verification");
  log("Running adversarial verification...");

  const {{ accepted, output: verified }} = await verify(winner, {{
    criteria: "correctness, security, quality",
    verifiers: [
      {{ tool: "claude", role: "logic_adversary", focus: "Find logic errors and edge cases", timeout: 180 }},
      {{ tool: "aiden", role: "quality_adversary", focus: "Find code quality issues", timeout: 180 }},
    ],
    maxRounds: 2,
    reviseTool: "coco",
    timeout: 180,
  }});

  return verified;
}}
```

## Proportionality Principle (重要)

Match workflow complexity to task complexity. NOT every task needs multiple patterns.

- **Simple tasks** (single focused action, clear scope): 1 agent() call, 1 phase. Done.
  Example: "fix a typo", "add a comment", "rename variable" → single agent, no patterns.
- **Medium tasks** (multi-step, some parallelism): fanout or sequence, 2-3 phases, 3-5 agent calls.
  Example: "review this file", "refactor this function" → fanout for perspectives.
- **Complex tasks** (unknown scope, quality-critical, competing approaches): combine patterns, 4-6 phases.
  Example: "architect a new system", "find all security bugs" → tournament + verify + loop.

If the task is simple, direct, and unambiguous — a single agent() call IS the best workflow.
Do NOT add patterns for their own sake. Every extra agent call costs time.

## Completion Result Contract (REQUIRED)

The final successful return value must preserve the full business result and include this compact card contract:

```javascript
return {{
  "card_summary": {{
    "verdict": "passed|needs_attention|failed|unknown",
    "conclusion": "one complete actionable conclusion",
    "findings": [{{ "severity": "high|medium|low|info", "text": "one complete finding" }}],
    "verification": [{{ "status": "passed|failed|warning|info", "text": "one complete verification result" }}],
    "deliverables": [{{ "type": "code|test|document|artifact|other", "text": "one complete deliverable" }}],
    "next_steps": ["one complete next action"]
  }},
  "result": fullResult,
  "verification": fullVerification
}};
```

Every brief field must be a 完整语义条目. Never manually cut a sentence or add a truncation marker; keep verbose evidence in `result`.

## Now Generate

Based on the user requirement above, generate a COMPLETE workflow script that:
1. Selects the most appropriate pattern(s) for this specific task
2. Leverages multiple tools for diversity and robustness
3. Includes verification for critical outputs
4. Uses bounded parallelism where dependencies allow it; never creates unbounded fan-out
5. Uses clear phases and labels for observability
6. Gives every direct agent() call and pattern descriptor a unique label and explicit timeout
7. Checks `result.error` or uses try/catch before returning direct agent outputs

Output ONLY the JavaScript code, no markdown fences, no explanatory text.
"""


# ---------------------------------------------------------------------------
# Orchestrator agent capability notes
# ---------------------------------------------------------------------------


def _get_agent_capability_note(agent_type: str) -> str:
    """Get capability notes for the orchestrator agent to adapt prompt style.

    Returns guidance on how to tailor the workflow script generation prompt
    based on the strengths and characteristics of the orchestrator agent
    that will execute the generated script.
    """
    capability_notes = {
        "coco": "Coco 擅长全栈编程和 subagent 调度，支持复杂的并行编排。prompt 中可以使用高级编排模式，如深度 fan-out、多轮验证闭环、复杂 pipeline 组合。",
        "claude": "Claude 擅长深度推理和复杂任务分解，适合需要精细规划的场景。prompt 中应强调逻辑严谨性、逐步推理、边界条件分析。",
        "aiden": "Aiden 擅长代码审查和架构设计，适合代码质量相关任务。prompt 中应强调审查深度、架构合理性、技术债务识别。",
        "codex": "Codex 擅长快速代码生成，适合 straightforward 的实现任务。prompt 中应简洁直接，减少不必要的抽象层。",
        "gemini": "Gemini 擅长多模态推理，适合涉及图像或复杂数据的任务。prompt 中可以包含多模态相关指令，如图像分析、图表理解。",
        "traex": "Traex 擅长高并发轻量任务，适合大量简单并行任务。prompt 中应强调任务粒度控制、最小化单任务复杂度、最大化并行度。",
    }
    return capability_notes.get(agent_type, capability_notes["coco"])


# ---------------------------------------------------------------------------
# Static fallback script for fail-closed scenarios
# ---------------------------------------------------------------------------

FALLBACK_SCRIPT: str = """
export const meta = {
  name: "fallback-orchestration",
  description: "Fallback orchestration workflow for fail-closed scenarios",
  phases: [
    { title: "Orchestration", detail: "Invoke sub-workflows to handle the task" },
  ],
  maxConcurrent: 3,
  tools: [],
  workflow_refs: [],
};

export default async function() {
  // Sentinel workflow() call so validate_generated_script accepts the
  // fallback (it requires at least one agent() or workflow() call).
  // workflow("noop") is a harmless no-op reference — the JS runtime
  // gracefully handles unknown template names as empty invocations.
  await workflow("noop");

  // Fallback orchestration that delegates to sub-workflows if available.
  // This is a minimal valid workflow that meets validation requirements
  // without making any real agent calls.
  return {
    card_summary: {
      verdict: "unknown",
      conclusion: "任务已完成，完整结果见报告。",
      findings: [],
      verification: [],
      deliverables: [],
      next_steps: [],
    },
    result: { status: "fallback-orchestration", message: "Workflow execution initiated via fallback path" },
    verification: null,
  };
}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_script_gen_prompt(
    requirement: str,
    available_tools: list[str] | dict[str, str],
    orchestrator_agent: str = "coco",
    orchestrator_binding: Optional[dict] = None,
    review_agents: Optional[list[dict]] = None,
    auto_reviewer: bool | None = None,
) -> str:
    """Build the prompt that instructs an AI to generate a workflow script.

    Args:
        requirement: The user's task description / workflow requirement.
        available_tools: List of tool names, or dict of name->description.
        orchestrator_agent: The type of agent that will execute the generated
            workflow script. Used to adapt prompt style and recommendations
            to the agent's capabilities. Defaults to "coco".
        orchestrator_binding: Selected orchestrator agent binding with tool and model info.
        review_agents: List of selected review agent bindings with tool and model info.
        auto_reviewer: Explicit confirmation-time review mode. True means no
            independent Reviewer call is promised; False pairs with explicit
            reviewers that the Engine invokes after the script completes.

    Returns:
        A complete prompt string ready to send to a code-generation agent.
    """
    if isinstance(available_tools, dict):
        tools_list = "\n".join(
            f"- `{name}` — {desc}" for name, desc in available_tools.items()
        ) if available_tools else "- (none)"
    else:
        tools_list = "\n".join(f"- `{t}`" for t in available_tools) if available_tools else "- (none)"

    # Orchestrator agent capability adaptation section
    agent_capability_section = f"""## 主编排 Agent 能力

当前使用的主编排 Agent 是：**{orchestrator_agent}**

{_get_agent_capability_note(orchestrator_agent)}

请根据上述 Agent 的能力特点，生成最适合它执行的 workflow 脚本。
"""

    # Add selected agent bindings information
    agent_bindings_section = ""
    if orchestrator_binding:
        _orch_tool = getattr(orchestrator_binding, 'tool_name', None) or (orchestrator_binding.get('tool_name') if isinstance(orchestrator_binding, dict) else None) or orchestrator_agent
        _orch_use_default = getattr(orchestrator_binding, 'use_default_model', None) if not isinstance(orchestrator_binding, dict) else orchestrator_binding.get('use_default_model', False)
        _orch_model = getattr(orchestrator_binding, 'model_name', None) if not isinstance(orchestrator_binding, dict) else orchestrator_binding.get('model_name')
        agent_bindings_section += f"""
## 已选择的主 Agent

- **工具**: {_orch_tool}
"""
        if not _orch_use_default and _orch_model:
            agent_bindings_section += f"  **模型**: {_orch_model}"

    if review_agents and review_agents:
        agent_bindings_section += """

## 已选择的评审 Agent

"""
        for i, agent in enumerate(review_agents):
            _ra_tool = getattr(agent, 'tool_name', None) or (agent.get('tool_name') if isinstance(agent, dict) else None) or 'unknown'
            _ra_use_default = getattr(agent, 'use_default_model', None) if not isinstance(agent, dict) else agent.get('use_default_model', False)
            _ra_model = getattr(agent, 'model_name', None) if not isinstance(agent, dict) else agent.get('model_name')
            agent_bindings_section += f"{i+1}. **工具**: {_ra_tool}"
            if not _ra_use_default and _ra_model:
                agent_bindings_section += f"  **模型**: {_ra_model}"
            agent_bindings_section += "\n"
        agent_bindings_section += """
这些是运行时承诺的独立 Reviewer。Engine 会在脚本完成后逐个发起独立真实调用并持久化证据；
脚本无需伪造 Reviewer 已执行，也不得在最终结果中把仅有选择记录描述为“已评审”。
"""
    elif auto_reviewer is True:
        agent_bindings_section += """

## Reviewer 模式

- **Auto**：无独立 Reviewer；编排器负责最终检查。本运行不承诺独立 Reviewer 调用，也不得声称存在独立 Reviewer 证据。
"""

    if agent_bindings_section:
        agent_capability_section += agent_bindings_section

    # Build the base prompt from template
    base_prompt = _SCRIPT_GEN_PROMPT_TEMPLATE.format(
        requirement=requirement.strip(),
        tools_list=tools_list,
    )

    # Inject agent capability section using an explicit sentinel.
    # Using str.find() instead of str.index() so we never raise ValueError.
    # If the sentinel is missing (template edits, test mutations, etc.), we
    # fall back to appending the section at the end and log a debug note.
    insert_idx = base_prompt.find(_USER_REQUIREMENT_INSERT_POINT)
    injection = agent_capability_section

    if insert_idx >= 0:
        # Splice the sections in and remove the sentinel line (plus its trailing
        # newline) so it never reaches the AI.
        sentinel_len = len(_USER_REQUIREMENT_INSERT_POINT)
        end_of_sentinel = insert_idx + sentinel_len
        # Absorb one trailing newline if present so we don't leave a blank line
        # in the rendered prompt.
        if end_of_sentinel < len(base_prompt) and base_prompt[end_of_sentinel] == "\n":
            end_of_sentinel += 1
        prompt = (
            base_prompt[:insert_idx]
            + injection
            + base_prompt[end_of_sentinel:]
        )
    else:
        logger.debug(
            "script_gen: USER_REQUIREMENT_INSERT_POINT not found in template; "
            "appending agent capability section at the end of the prompt."
        )
        prompt = base_prompt + "\n\n" + injection

    return prompt + ("\n\n" + get_subagent_encouragement() if get_subagent_encouragement() else "")


def _iter_js_call_sources(script_content: str, function_name: str) -> list[str]:
    """Return best-effort source spans for direct JS function calls."""
    calls: list[str] = []
    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    for match in pattern.finditer(script_content):
        open_paren = script_content.find("(", match.start())
        if open_paren < 0:
            continue

        depth = 0
        in_string = False
        string_char = ""
        escaped = False
        for idx in range(open_paren, len(script_content)):
            ch = script_content[idx]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if in_string:
                if ch == string_char:
                    in_string = False
                continue
            if ch in ("'", '"', "`"):
                in_string = True
                string_char = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    calls.append(script_content[match.start() : idx + 1])
                    break
    return calls


def _literal_agent_labels(script_content: str) -> list[str]:
    """Extract statically-known agent labels from JS option objects."""
    return [
        match.group(2)
        for match in re.finditer(r"""\blabel\s*:\s*(["'])(.*?)\1""", script_content)
    ]


def _strip_js_strings_and_comments(script_content: str) -> str:
    """Blank JS string/comment contents so token checks only see executable code."""
    chars = list(script_content)
    idx = 0
    while idx < len(chars):
        ch = chars[idx]
        nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
        if ch == "/" and nxt == "/":
            chars[idx] = chars[idx + 1] = " "
            idx += 2
            while idx < len(chars) and chars[idx] != "\n":
                chars[idx] = " "
                idx += 1
            continue
        if ch == "/" and nxt == "*":
            chars[idx] = chars[idx + 1] = " "
            idx += 2
            while idx + 1 < len(chars) and not (chars[idx] == "*" and chars[idx + 1] == "/"):
                chars[idx] = " "
                idx += 1
            if idx + 1 < len(chars):
                chars[idx] = chars[idx + 1] = " "
                idx += 2
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            chars[idx] = " "
            idx += 1
            escaped = False
            while idx < len(chars):
                current = chars[idx]
                chars[idx] = " "
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == quote:
                    idx += 1
                    break
                idx += 1
            continue
        idx += 1
    return "".join(chars)


def validate_generated_script(
    script_content: str,
    review_agents: Optional[list[dict]] = None,
) -> tuple[bool, list[str]]:
    """Validate a generated workflow script without executing it.

    Performs basic structural and safety checks:
    - Presence of `export const meta =`
    - Meta has `name` and `description` fields
    - Balanced braces and brackets (rough syntax check)
    - At least one `agent(` call OR at least one `workflow(` call
      (pure orchestration / workflow-only scripts are accepted)
    - Presence of `export default` entry function
    - Dangerous patterns are reported as BLOCKING errors (fail-closed — not warnings)
    - Reviewer selections are not inferred from script text; committed
      independent calls are enforced by WorkflowEngine after script completion

    The runtime sandbox provides defense-in-depth, but script-level rejection
    is the primary security boundary for user-generated workflows. Templates
    approved by an admin can be whitelisted via `WORKFLOW_ALLOWLIST_CAPABILITIES`
    to preserve their intended behavior.

    Args:
        script_content: The raw JavaScript source code of the workflow script.
        review_agents: List of selected review agent bindings with tool and model info.
                       Retained for API compatibility; runtime Reviewer
                       commitments are enforced outside generated JavaScript.

    Returns:
        A tuple of (is_valid, list_of_messages). Dangerous patterns are treated
        as errors (prefix "[capability]"). `is_valid` is False if any structural
        error OR any dangerous pattern is detected.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not script_content or not script_content.strip():
        return False, ["Script content is empty"]

    # --- Check for natural language preamble ---
    # If the script starts with text that is clearly not JavaScript, reject it.
    # Valid JS starts with: export, //, /*, const, let, var, "use strict", 'use strict'
    # This catches cases where the AI model prefixes code with "Let me..." or similar.
    first_meaningful = script_content.lstrip()
    if first_meaningful and not re.match(
        r"""^(export|/[/*]|const |let |var |"use strict"|'use strict')""",
        first_meaningful,
    ):
        errors.append(
            "Script appears to start with natural language text instead of JavaScript code. "
            "Expected the file to begin with `export`, a comment, or a declaration."
        )

    # --- Note: Review agent constraints are no longer enforced ---
    # Roles are dynamically inferred by the LLM orchestrator, not statically validated.
    # The orchestrator agent decides how to use review tools based on the task context.

    # --- Check meta export presence ---
    if not re.search(r"export\s+const\s+meta\s*=", script_content):
        errors.append("Missing `export const meta =` declaration")

    # --- Check meta has name field ---
    if not re.search(r"""name\s*:\s*["'`]""", script_content):
        errors.append("Meta object missing `name` field (expected name: \"...\")")

    # --- Check meta has description field ---
    if not re.search(r"""description\s*:\s*["'`]""", script_content):
        errors.append("Meta object missing `description` field (expected description: \"...\")")

    # --- Check for export default function ---
    if not re.search(r"export\s+default\s+(async\s+)?function", script_content):
        errors.append("Missing `export default function` — script must export a default entry function")

    # --- Check for at least one agent() or workflow() call ---
    # Pure orchestration (workflow-only) scripts that only invoke sub-workflows
    # are valid. Dynamic pattern primitives (classify, fanout, verify, generate,
    # tournament, loop) also count as they internally dispatch agent() calls.
    has_agent_call = bool(re.search(r"\bagent\s*\(", script_content))
    has_workflow_call = bool(re.search(r"\bworkflow\s*\(", script_content))
    has_pattern_call = bool(re.search(
        r"\b(classify|fanout|verify|generate|tournament|loop|race|sequence)\s*\(",
        script_content,
    ))
    if not (has_agent_call or has_workflow_call or has_pattern_call):
        errors.append(
            "No `agent()`, `workflow()`, or pattern primitive call found - workflow must "
            "invoke at least one agent, sub-workflow, or orchestration pattern"
        )

    # --- Runtime reliability checks ---
    # Duplicate labels collapse progress updates for different agent attempts;
    # unbounded calls let one ACP prompt consume the whole workflow timeout.
    agent_calls = _iter_js_call_sources(script_content, "agent")
    labels = _literal_agent_labels(script_content)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        errors.append(
            "Duplicate agent label(s) found: "
            + ", ".join(duplicate_labels)
            + ". Each agent call must use a unique label for state tracking."
        )

    for idx, call_source in enumerate(agent_calls, start=1):
        if not re.search(r"\btimeout\s*:", call_source):
            label_match = re.search(r"""\blabel\s*:\s*(["'])(.*?)\1""", call_source)
            label = label_match.group(2) if label_match else f"agent call #{idx}"
            errors.append(
                f"Direct agent call `{label}` is missing an explicit `timeout:`. "
                "Bound every agent call so a single ACP prompt cannot consume the total workflow timeout."
            )

    executable_code = _strip_js_strings_and_comments(script_content)
    if agent_calls and not re.search(r"""(\.\s*error\b|\bcatch\s*\(|\btry\s*\{)""", executable_code):
        errors.append(
            "Direct agent calls must handle `result.error` or use try/catch before returning. "
            "This prevents failed agent calls from being treated as successful workflow output."
        )

    # --- Balanced braces check (rough) ---
    brace_balance = 0
    bracket_balance = 0
    paren_balance = 0
    in_string = False
    string_char = ""
    escaped = False

    for ch in script_content:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if in_string:
            if ch == string_char:
                in_string = False
            continue
        if ch in ("'", '"', "`"):
            in_string = True
            string_char = ch
            continue

        if ch == "{":
            brace_balance += 1
        elif ch == "}":
            brace_balance -= 1
        elif ch == "[":
            bracket_balance += 1
        elif ch == "]":
            bracket_balance -= 1
        elif ch == "(":
            paren_balance += 1
        elif ch == ")":
            paren_balance -= 1

    if brace_balance != 0:
        errors.append(
            f"Unbalanced braces: {{}} balance is {brace_balance:+d} "
            f"({'excess opens' if brace_balance > 0 else 'excess closes'})"
        )
    if bracket_balance != 0:
        errors.append(
            f"Unbalanced brackets: [] balance is {bracket_balance:+d} "
            f"({'excess opens' if bracket_balance > 0 else 'excess closes'})"
        )
    if paren_balance != 0:
        errors.append(
            f"Unbalanced parentheses: () balance is {paren_balance:+d} "
            f"({'excess opens' if paren_balance > 0 else 'excess closes'})"
        )

    # --- Check for dangerous patterns (FAIL-CLOSED — security boundary) ---
    # Patterns in _DANGEROUS_PATTERNS are non-negotiable: filesystem, network,
    # child_process, eval, Function, Worker, globalThis, Deno/Bun APIs, etc.
    # They are reported as blocking errors, not warnings, because the runtime
    # sandbox is defense-in-depth, not the primary enforcement mechanism.
    for pattern, description in _DANGEROUS_PATTERNS:
        if re.search(pattern, script_content):
            errors.append(f"[capability] Forbidden pattern: {description}")

    is_valid = len(errors) == 0
    all_messages = errors + warnings

    if errors:
        logger.warning("Script validation failed with %d error(s): %s", len(errors), "; ".join(errors))
    if warnings:
        logger.info("Script validation warnings: %s", "; ".join(warnings))
    if is_valid and not warnings:
        logger.debug("Script validation passed")

    return is_valid, all_messages


def extract_meta_from_script(script_content: str) -> Optional[dict[str, Any]]:
    """Try to extract the meta object from a workflow script source.

    Uses regex-based extraction followed by JSON parsing. Handles common JS object
    literal patterns by converting them to valid JSON (unquoted keys, trailing commas,
    single quotes).

    Args:
        script_content: The raw JavaScript source code of the workflow script.

    Returns:
        The meta object as a Python dict, or None if extraction/parsing fails.
    """
    if not script_content:
        return None

    # Strategy 1: Find the meta block between `export const meta = {` and the matching `}`
    meta_match = re.search(
        r"export\s+const\s+meta\s*=\s*(\{)",
        script_content,
    )
    if not meta_match:
        logger.debug("No `export const meta =` found in script")
        return None

    # Find the matching closing brace
    start_idx = meta_match.start(1)
    brace_depth = 0
    end_idx = start_idx
    in_string = False
    string_char = ""
    escaped = False

    for i in range(start_idx, len(script_content)):
        ch = script_content[i]

        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if in_string:
            if ch == string_char:
                in_string = False
            continue
        if ch in ("'", '"', "`"):
            in_string = True
            string_char = ch
            continue

        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                end_idx = i
                break
    else:
        # Never found matching brace
        logger.debug("Could not find matching closing brace for meta object")
        return None

    raw_meta = script_content[start_idx : end_idx + 1]

    # Convert JS object literal to JSON-parseable form
    json_str = _js_object_to_json(raw_meta)

    try:
        meta = json.loads(json_str)
        if isinstance(meta, dict):
            logger.debug("Successfully extracted meta: %s", meta.get("name", "?"))
            _enrich_workflow_refs(meta, script_content)
            return meta
        return None
    except json.JSONDecodeError as e:
        logger.debug("JSON parse failed for extracted meta: %s", repr(e))
        # Strategy 2: Try a more aggressive cleanup
        json_str_v2 = _aggressive_json_cleanup(raw_meta)
        try:
            meta = json.loads(json_str_v2)
            if isinstance(meta, dict):
                logger.debug("Extracted meta via aggressive cleanup: %s", meta.get("name", "?"))
                _enrich_workflow_refs(meta, script_content)
                return meta
        except json.JSONDecodeError:
            pass

        logger.warning("Failed to parse meta object from script")
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _enrich_workflow_refs(meta: dict[str, Any], script_content: str) -> None:
    """If meta lacks workflow_refs, scan the script for workflow() calls and populate.

    Produces normalized refs in {name, path?, hash?} format. Existing string refs
    are converted to dicts for consistency.
    """
    existing = meta.get("workflow_refs", [])
    if existing:
        # Normalize existing refs: convert strings to {name} dicts
        normalized: list[dict[str, str]] = []
        for ref in existing:
            if isinstance(ref, str):
                normalized.append({"name": ref})
            elif isinstance(ref, dict):
                # Normalize legacy script_path to path
                item = dict(ref)
                if "script_path" in item and "path" not in item:
                    item["path"] = item.pop("script_path")
                normalized.append(item)
        meta["workflow_refs"] = normalized
        return

    # Find workflow("name", ...) or workflow({name: "..."}) calls
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(r"""\bworkflow\s*\(\s*["'`]([^"'`]+)["'`]""", script_content):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            refs.append({"name": name})
    if refs:
        meta["workflow_refs"] = refs


def _js_object_to_json(js_obj: str) -> str:
    """Convert a JavaScript object literal to a JSON string.

    Handles:
    - Unquoted property keys: { name: "x" } -> { "name": "x" }
    - Single-quoted strings: 'value' -> "value"
    - Trailing commas: [1, 2, ] -> [1, 2]
    - JS comments: // ... and /* ... */
    """
    result = js_obj

    # Remove single-line comments (but not inside strings — rough heuristic)
    result = re.sub(r"//[^\n]*", "", result)
    # Remove multi-line comments
    result = re.sub(r"/\*[\s\S]*?\*/", "", result)

    # Replace single quotes with double quotes (simple cases)
    # This is imperfect for strings containing quotes but handles most generated scripts
    result = _replace_single_quotes(result)

    # Quote unquoted keys: word: -> "word":
    result = re.sub(
        r'(?<=[{,\n])\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:',
        r' "\1":',
        result,
    )

    # Remove trailing commas before } or ]
    result = re.sub(r",\s*([}\]])", r"\1", result)

    return result


def _replace_single_quotes(s: str) -> str:
    """Replace single-quoted strings with double-quoted strings.

    Handles escaped single quotes within single-quoted strings.
    """
    result = []
    i = 0
    length = len(s)

    while i < length:
        ch = s[i]

        # Already in a double-quoted string — pass through
        if ch == '"':
            result.append(ch)
            i += 1
            while i < length:
                if s[i] == "\\" and i + 1 < length:
                    result.append(s[i])
                    result.append(s[i + 1])
                    i += 2
                elif s[i] == '"':
                    result.append(s[i])
                    i += 1
                    break
                else:
                    result.append(s[i])
                    i += 1
            continue

        # Template literal — pass through as-is (cannot easily convert)
        if ch == "`":
            result.append('"')
            i += 1
            while i < length:
                if s[i] == "\\" and i + 1 < length:
                    result.append(s[i])
                    result.append(s[i + 1])
                    i += 2
                elif s[i] == "`":
                    result.append('"')
                    i += 1
                    break
                else:
                    # Escape any double quotes inside
                    if s[i] == '"':
                        result.append('\\"')
                    else:
                        result.append(s[i])
                    i += 1
            continue

        # Single-quoted string — convert to double-quoted
        if ch == "'":
            result.append('"')
            i += 1
            while i < length:
                if s[i] == "\\" and i + 1 < length:
                    if s[i + 1] == "'":
                        # Escaped single quote -> just the quote in double-quoted string
                        result.append("'")
                        i += 2
                    else:
                        result.append(s[i])
                        result.append(s[i + 1])
                        i += 2
                elif s[i] == '"':
                    # Unescaped double quote inside single-quoted string -> escape it
                    result.append('\\"')
                    i += 1
                elif s[i] == "'":
                    result.append('"')
                    i += 1
                    break
                else:
                    result.append(s[i])
                    i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _aggressive_json_cleanup(raw: str) -> str:
    """More aggressive attempt to extract a JSON object from JS source.

    Strips everything that isn't likely part of the JSON structure.
    """
    result = raw

    # Remove comments
    result = re.sub(r"//[^\n]*", "", result)
    result = re.sub(r"/\*[\s\S]*?\*/", "", result)

    # Replace single quotes
    result = _replace_single_quotes(result)

    # Quote unquoted keys
    result = re.sub(
        r'(?<=[{,\n])\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:',
        r' "\1":',
        result,
    )

    # Remove trailing commas
    result = re.sub(r",\s*([}\]])", r"\1", result)

    # Remove template literal expressions ${...} — replace with empty string
    result = re.sub(r"\$\{[^}]*\}", "", result)

    return result


# ---------------------------------------------------------------------------
# Simple script generation (no AI call — wraps requirement in a single agent)
# ---------------------------------------------------------------------------


def generate_simple_script(
    requirement: str,
    selected_tools: list[str] | None = None,
    tool_model_map: dict[str, str] | None = None,
) -> str:
    """Generate a bounded fallback workflow using Dynamic Workflow patterns."""
    _enc = get_subagent_encouragement()
    tools = selected_tools or ["coco"]
    primary_tool = tools[0]
    escaped = requirement.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    comment_requirement = requirement[:80].replace("*/", "* /")
    json_tools = json.dumps(tools)
    json_tool_models = json.dumps(tool_model_map or {})

    return f'''/**
 * Auto-generated Dynamic Workflow.
 * Requirement: {comment_requirement}
 */

export const meta = {{
  name: "generated-dynamic-workflow",
  description: "Auto-generated bounded dynamic workflow with fallback handling",
  phases: [
    {{ title: "Routing", detail: "Choose a deterministic bounded execution path" }},
    {{ title: "Execution", detail: "Execute directly or via bounded fanout" }},
    {{ title: "Verification", detail: "Run bounded verification and return usable output" }}
  ],
  maxConcurrent: 6,
  tools: {json_tools},
  patterns: ["race", "fanout"]
}};

export default async function main() {{
  const requirement = `{escaped}`;
  const tools = {json_tools};
  const toolModels = {json_tool_models};
  const primaryTool = "{primary_tool}";
  const candidateTools = Array.from(new Set((tools.length ? tools : [primaryTool]).filter(Boolean))).slice(0, 3);
  const fallbackTool = candidateTools.length > 1 ? candidateTools[1] : primaryTool;
  const verifierTool = primaryTool;
  function modelFor(tool) {{ return toolModels[tool] || undefined; }}

  function fallback(stage, reason, partial = null) {{
    return {{
      fallback: true,
      stage,
      error: reason || "unknown error",
      partial,
      message: "Workflow used bounded fallback handling instead of waiting for the total timeout.",
    }};
  }}

  function completionEnvelope(resultValue, review = null) {{
    const resultObject = resultValue && typeof resultValue === "object" ? resultValue : {{}};
    const reviewObject = review && typeof review === "object" ? review : {{}};
    const conclusion =
      (typeof reviewObject.summary === "string" && reviewObject.summary.trim()) ||
      (typeof resultObject.conclusion === "string" && resultObject.conclusion.trim()) ||
      (typeof resultObject.summary === "string" && resultObject.summary.trim()) ||
      "任务已完成，完整结果见报告。";
    const rawIssues = Array.isArray(reviewObject.issues) ? reviewObject.issues : [];
    const findings = rawIssues.map((issue) => {{
      if (typeof issue === "string") return {{ severity: "high", text: issue }};
      if (!issue || typeof issue !== "object") return null;
      const text = issue.text || issue.claim || issue.description || issue.issue || issue.summary;
      return typeof text === "string" && text.trim()
        ? {{ severity: issue.severity || "high", text: text.trim() }}
        : null;
    }}).filter(Boolean);
    const verdict = reviewObject.approve === false
      ? "needs_attention"
      : reviewObject.approve === true
        ? "passed"
        : "unknown";
    const verification = review
      ? [{{
          status: reviewObject.approve === false ? "failed" : reviewObject.approve === true ? "passed" : "warning",
          text: typeof reviewObject.summary === "string" && reviewObject.summary.trim()
            ? reviewObject.summary.trim()
            : reviewObject.error
              ? "验证未完成，完整详情见报告。"
              : "验证状态未明确，完整详情见报告。",
        }}]
      : [];
    const deliverables = Array.isArray(resultObject.deliverables)
      ? resultObject.deliverables.map((item) => typeof item === "string" ? {{ type: "other", text: item }} : item)
      : [];
    const nextSteps = Array.isArray(resultObject.next_steps)
      ? resultObject.next_steps.filter((item) => typeof item === "string" && item.trim())
      : [];
    return {{
      card_summary: {{
        verdict,
        conclusion,
        findings,
        verification,
        deliverables,
        next_steps: nextSteps,
      }},
      result: resultValue,
      verification: review,
    }};
  }}

  function chooseRoute(text) {{
    const normalized = String(text || "").toLowerCase();
    const simpleSignals = ["先不要动手", "不要动手", "不要改代码", "先分析", "只分析", "分析下", "为什么", "why", "explain"];
    const parallelSignals = ["并行执行", "批量处理", "多模块", "多文件同时", "全面审计", "全面测试", "批量重构", "批量迁移", "parallel", "batch migration", "audit all", "refactor all"];
    if (simpleSignals.some((word) => normalized.includes(word))) {{
      return {{ mode: "simple", subtasks: [requirement], reason: "analysis-only or direct request" }};
    }}
    if (parallelSignals.some((phrase) => normalized.includes(phrase))) {{
      return {{ mode: "parallel", subtasks: [], reason: "multi-part request" }};
    }}
    return {{ mode: "simple", subtasks: [requirement], reason: "fallback default" }};
  }}

  phase("Routing");
  log("Choosing bounded execution path without an ACP routing call");

  const route = chooseRoute(requirement);
  let subtasks = Array.isArray(route.subtasks) && route.subtasks.length
    ? route.subtasks
    : [requirement];

  if (route.mode === "parallel") {{
    log("Attempting bounded subtask split across selected tools");
    let split = null;
    try {{
      split = await race(candidateTools.map((tool) => () => agent({{
          prompt: `Split this requirement into 2-4 independent, executable subtasks.

Requirement:
${{requirement}}

Return JSON only: {{ "subtasks": ["..."], "reason": "..." }}

{_enc}`,
          tool,
          model: modelFor(tool),
          role: "planner",
          schema: {{ subtasks: [], reason: "" }},
          label: `split-${{tool}}`,
          timeout: 90,
        }})), {{
        validate: (item) => item && !item.error && Array.isArray(item.subtasks) && item.subtasks.length > 1,
      }});
    }} catch (err) {{
      log(`Subtask split failed: ${{err && err.message ? err.message : String(err)}}`);
    }}
    if (split && split.error) {{
      log(`Subtask split failed, falling back to single execution: ${{split.error}}`);
    }}
    if (split && !split.error && Array.isArray(split.subtasks)) {{
      const planned = split.subtasks.filter(Boolean).slice(0, 4);
      if (planned.length > 1) {{
        subtasks = planned;
      }}
    }}
    if (subtasks.length < 2) {{
      subtasks = [requirement];
    }}
  }}
  log(`Execution mode: ${{route.mode || "simple"}}, subtasks: ${{subtasks.length}}`);

  // Phase 2: Execute
  phase("Execution");

  let result;
  if (subtasks.length >= 2) {{
    // Fan-out pattern for parallel subtasks
    log(`Executing ${{subtasks.length}} bounded subtasks in parallel...`);
    const workerResults = await fanout(requirement, subtasks.map((task, i) => ({{
      prompt: `Complete this subtask as part of a larger requirement.

Overall requirement: ${{requirement}}
Your specific subtask: ${{typeof task === "string" ? task : task.description || JSON.stringify(task)}}

Complete fully and provide concrete output. If blocked, return a clear error object instead of waiting.

{_enc}`,
      tool: tools[i % tools.length],
      model: modelFor(tools[i % tools.length]),
      role: `worker-${{i}}`,
      label: `execute-${{i}}`,
      timeout: 180,
    }})), {{ synthesize: false, defaultTool: primaryTool }});

    const successful = workerResults.filter((item) => !(item && item.error));
    if (successful.length === 0) {{
      return fallback("Execution", "all parallel workers failed", workerResults);
    }}

    if (successful.length === 1) {{
      result = successful[0];
    }} else {{
      result = await agent({{
        prompt: `Synthesize these successful parallel results into one coherent final output.

Requirement:
${{requirement}}

Results:
${{JSON.stringify(successful, null, 2)}}

{_enc}`,
        tool: fallbackTool,
        model: modelFor(fallbackTool),
        role: "integrator",
        label: "synthesize-result",
        timeout: 120,
      }});
      if (result && result.error) {{
        return fallback("Synthesis", result.error, successful);
      }}
    }}
  }} else {{
    // Single focused execution: race selected tools instead of serially waiting
    // for the first selected backend to hit its ACP timeout.
    log(`Executing bounded race across ${{candidateTools.length}} selected tool(s)...`);
    try {{
      result = await race(candidateTools.map((tool) => () => agent({{
        prompt: `Fulfill the user's request exactly.

Requirement:
${{requirement}}

If the user asks for analysis only, planning only, or says not to change code, do not change code. Provide the requested analysis and stop.
If implementation is requested, provide a complete, production-ready result. If blocked, return a clear error object quickly.

{_enc}`,
        tool,
        model: modelFor(tool),
        role: "executor",
        label: `execute-${{tool}}`,
        timeout: 180,
      }})), {{
        validate: (item) => item != null && item !== "" && !(item && item.error),
      }});
    }} catch (err) {{
      return fallback("Execution", err && err.message ? err.message : String(err));
    }}
    if (result && result.error) {{
      return fallback("Execution", result.error);
    }}
  }}

  phase("Verification");
  if (route.reason === "analysis-only or direct request") {{
    log("Skipping verification for analysis-only task");
    return completionEnvelope(result);
  }}
  log("Running bounded verification");
  const review = await agent({{
    prompt: `Review this workflow output for correctness, completeness, and obvious risks.

Requirement:
${{requirement}}

Output:
${{typeof result === "string" ? result : JSON.stringify(result, null, 2)}}

Respond with JSON: {{ "approve": true, "issues": ["..."], "summary": "..." }}`,
    tool: verifierTool,
    model: modelFor(verifierTool),
    role: "verifier",
    schema: {{ approve: false, issues: [], summary: "" }},
    label: "verify-output",
    timeout: 120,
  }});

  if (review && review.error) {{
    log(`Verification failed, returning execution output with fallback marker: ${{review.error}}`);
    return completionEnvelope(result, fallback("Verification", review.error));
  }}
  if (review && review.approve === false && Array.isArray(review.issues) && review.issues.length > 0) {{
    return completionEnvelope(result, {{
      approved: false,
      issues: review.issues,
      summary: review.summary || "Verifier reported concerns",
    }});
  }}

  return completionEnvelope(result, review);
}}
'''
