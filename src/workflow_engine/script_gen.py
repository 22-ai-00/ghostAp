"""Generate and validate sandboxed Dynamic Workflow scripts."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUBAGENT_ENCOURAGEMENT = (
    "**Bounded Delegation**: Handle a focused task directly. Delegate only genuinely "
    "independent work for a medium or complex task, keep the fan-out bounded, and do "
    "not ask nested subagents to repeat orchestration already owned by this workflow."
)


def _subagent_hint_enabled() -> bool:
    try:
        from src.config import get_settings

        return bool(getattr(get_settings(), "workflow_subagent_hint_enabled", True))
    except Exception:
        return True


def get_subagent_encouragement() -> str:
    return SUBAGENT_ENCOURAGEMENT if _subagent_hint_enabled() else ""


# Source validation is the primary boundary; the Node sandbox is defense in depth.
_DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
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
    (
        r"""\.constructor\s*\.\s*constructor\s*\(""",
        "constructor.constructor escape (reaching the host Function constructor)",
    ),
    (r"""new\s+Worker\s*\(""", "Worker thread creation"),
    (r"""globalThis\[""", "globalThis bracket access"),
    (r"""Deno\.""", "Deno runtime API"),
    (r"""Bun\.""", "Bun runtime API"),
    (r"""\bimport\s*\(""", "dynamic import() expression"),
    (r"""\bimport\.meta\b""", "import.meta access"),
)


_CAPABILITY_NOTES = {
    "coco": "Coco 擅长全栈编程、subagent 调度和复杂并行编排。",
    "claude": "Claude 擅长深度推理；强调逻辑严谨性和边界条件。",
    "aiden": "Aiden 擅长代码审查和架构设计。",
    "codex": "Codex 擅长快速代码生成；指令应简洁直接。",
    "gemini": "Gemini 擅长多模态推理和图像理解。",
    "traex": "Traex 擅长高并发轻量任务。",
}


def _get_agent_capability_note(agent_type: str) -> str:
    return _CAPABILITY_NOTES.get(agent_type, _CAPABILITY_NOTES["coco"])


_SCRIPT_GEN_PROMPT_TEMPLATE = """# Workflow Script Generation Task

## User Requirement

<<REQUIREMENT>>

## Available Resources

**Tools (AI agents you can dispatch):**
<<TOOLS>>

**Roles (specialized perspectives for agents):**
根据任务需求自行规划角色分工。每个 agent() 调用可通过 `role` 参数指定适合的角色，例如 architect、reviewer、tester 等。
角色不是固定列表。建议考虑：架构设计、代码实现、安全审计、正确性验证、测试覆盖等维度。

<<RUNTIME_BINDING>>

## Dynamic Workflow Algorithm

脚本是本次任务的可执行计划。运行时负责确定性控制流，Agent 只负责语义工作；整个主路径自动推进，不得要求用户选择 Agent、确认脚本、批准继续或手动恢复。

按 **Scope -> Pipeline -> Verify -> Synthesize** 组织：

1. **Scope**：范围明确时直接生成有上限的 worklist；范围未知时只派一个轻量 scout 发现并裁剪工作项。不要先派一个大而慢的 analysis agent。
2. **Pipeline**：同构 work item 经历相同阶段时优先 `pipeline()`，保持 item 内串行、item 间有界并行。只有下游必须看到全部上游结果时才设置 barrier。
3. **Verify**：只对高风险或关键结果使用独立、结构化、有界验证。验证不确定、shape 失配或暂时失败时自动修复或重试，耗尽后返回明确失败，不等待用户。
4. **Synthesize**：过滤失败项，说明缺失范围，返回完整业务结果和紧凑卡片摘要；禁止只返回中间数组或把部分结果冒充全部成功。

### Proportionality

- 简单任务：1 phase、1 次 `agent()`，不额外路由或评审。
- 中等任务：`fanout`、`sequence` 或 `pipeline`，通常 3-5 次调用。
- 复杂任务：按依赖组合多个原语和 4-6 个 phase；禁止为展示编排而过度调用。

### 6+2 Dynamic Primitives

- `classify(input, categories, opts)`：分类后路由。
- `fanout(input, workers, opts)`：独立任务有界并行并可合成。
- `verify(output, opts)`：对抗验证与有界修订。
- `generate(count, generatorFn, filterFn, opts)`：生成候选并过滤；`count <= 50`。
- `tournament(contestants, judgeFn, opts)`：淘汰比较候选。
- `loop(taskFn, opts)`：迭代至收敛；`maxIterations <= 50`。
- `sequence(steps)`：严格顺序传递结果。
- `race(contestants, opts)`：只在多种独立方法都可能成功时取首个有效结果。

辅助原语：`pipeline(items, ...stages, opts)`、`parallel(functions)`、`phase(title)`、`log(message)`。直接基于用户需求选择 classify/fanout/verify/loop/race，不要生成固定的 Analysis -> Execution 模板。

### Agent Contract

```javascript
const result = await agent({
  prompt: "one focused task",
  tool: "one available tool",
  model: "optional bound model",
  role: "task-specific role",
  label: "unique-observable-label",
  schema: { summary: "", findings: [{ severity: "", text: "" }] },
  timeout: 180,
});
if (result && result.error) return { error: result.error, stage: "named-stage" };
```

- 每个 agent() label 必须唯一；不要复用 task-analysis、analysis 或 worker 等通用 label。
- 为每个 agent() 显式设置短超时：路由 60-90s，普通任务 120-180s，确需长推理才使用 300s。
- 检查 result.error 并提供 fallback；并行结果逐项过滤错误，全部失败时返回结构化失败。
- 跨节点数据使用紧凑递归 `schema`；shape 失配不得继续传给下游。
- 角色和工具按任务动态分配，但只能使用上方可用工具与运行时绑定。
- 总 Agent 调用（含原语内部调用）不得超过 200；并发必须有上限。
- 慢操作前和阶段里程碑调用 `log()`；不要记录完整 prompt、完整结果或敏感信息。

## Output Format

只输出完整 ES Module JavaScript，不要 Markdown fence 或解释。必须包含：

```javascript
export const meta = {
  name: "task-specific-kebab-name",
  description: "one-line description",
  phases: [{ title: "Scope", detail: "Bound the work" }],
  maxConcurrent: 6,
  tools: ["only-available-tools"],
  patterns: ["used-primitives"],
};

export default async function main() {
  // automatic bounded orchestration
  return {
    "card_summary": {
      "verdict": "passed|needs_attention|failed|unknown",
      "conclusion": "one complete actionable conclusion",
      "findings": [{ "severity": "high|medium|low|info", "text": "one complete finding" }],
      "verification": [{ "status": "passed|failed|warning|info", "text": "one complete verification result" }],
      "deliverables": [{ "type": "code|test|document|artifact|other", "text": "one complete deliverable" }],
      "next_steps": ["one complete next action"]
    },
    "result": fullResult,
    "verification": fullVerification
  };
}
```

每个摘要字段必须是完整语义条目；详细证据全部保留在 `result`，不得截断或返回 legacy 裸数组。

## Safety and Completion Rules

- 禁止 `require`、`import`、filesystem、network、child process、`process`、`eval`、`Function`、Worker 或 sandbox escape。
- 所有逻辑在单文件内；原语是全局变量，无需导入。
- 普通、安全、可逆选择采用推荐项；高风险且未获原始请求精确授权的动作拒绝或跳过，并继续安全部分。
- 用户主动 stop/cancel 才终止；其他提问、格式修复、Review 不确定和暂时失败均有界自动恢复。
- 不得声称未实际执行的独立 Reviewer、测试或验证已完成。

<<ENCOURAGEMENT>>
"""


def _binding_field(binding: Any, name: str, default: Any = None) -> Any:
    if binding is None:
        return default
    if isinstance(binding, dict):
        return binding.get(name, default)
    return getattr(binding, name, default)


def _binding_description(binding: Any, fallback_tool: str) -> str:
    tool = _binding_field(binding, "tool_name", fallback_tool) or fallback_tool
    model = _binding_field(binding, "model_name")
    use_default = bool(_binding_field(binding, "use_default_model", not model))
    return f"`{tool}` / " + ("backend default model" if use_default or not model else f"`{model}`")


def build_script_gen_prompt(
    requirement: str,
    available_tools: list[str] | dict[str, str],
    orchestrator_agent: str = "coco",
    orchestrator_binding: Optional[dict] = None,
    review_agents: Optional[list[dict]] = None,
    auto_reviewer: bool | None = None,
) -> str:
    """Build the compact, fully automatic Dynamic Workflow generation contract."""
    if isinstance(available_tools, dict):
        tools = "\n".join(f"- `{name}` - {desc}" for name, desc in available_tools.items())
    else:
        tools = "\n".join(f"- `{name}`" for name in available_tools)

    runtime = [
        "## Automatic Runtime Binding",
        f"- Orchestrator: {_binding_description(orchestrator_binding, orchestrator_agent)}",
        f"- Capability: {_get_agent_capability_note(orchestrator_agent)}",
    ]
    if review_agents:
        runtime.append("- Independent reviewers scheduled automatically by the runtime:")
        runtime.extend(
            f"  - {_binding_description(binding, 'unknown')}" for binding in review_agents
        )
        runtime.append("  Only their completed calls count as independent review evidence.")
    elif auto_reviewer:
        runtime.append(
            "- Review: Auto self-review by the orchestrator; no independent Reviewer evidence is promised."
        )
    else:
        runtime.append("- Review: apply proportional in-script verification without user interaction.")

    replacements = {
        "<<REQUIREMENT>>": requirement.strip(),
        "<<TOOLS>>": tools or "- (none registered)",
        "<<RUNTIME_BINDING>>": "\n".join(runtime),
        "<<ENCOURAGEMENT>>": get_subagent_encouragement(),
    }
    prompt = _SCRIPT_GEN_PROMPT_TEMPLATE
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt.strip()


def _strip_js_strings_and_comments(script_content: str) -> str:
    """Blank strings and comments while preserving source offsets."""
    chars = list(script_content)
    idx = 0
    while idx < len(chars):
        ch = chars[idx]
        nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
        if ch == "/" and nxt in ("/", "*"):
            block = nxt == "*"
            chars[idx] = chars[idx + 1] = " "
            idx += 2
            while idx < len(chars):
                if block and idx + 1 < len(chars) and chars[idx : idx + 2] == ["*", "/"]:
                    chars[idx] = chars[idx + 1] = " "
                    idx += 2
                    break
                if not block and chars[idx] == "\n":
                    break
                chars[idx] = " "
                idx += 1
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


def _iter_js_call_sources(script_content: str, function_name: str) -> list[str]:
    """Return best-effort source spans for executable direct JS calls."""
    masked = _strip_js_strings_and_comments(script_content)
    calls: list[str] = []
    for match in re.finditer(rf"\b{re.escape(function_name)}\s*\(", masked):
        open_paren = masked.find("(", match.start())
        depth = 0
        for idx in range(open_paren, len(masked)):
            if masked[idx] == "(":
                depth += 1
            elif masked[idx] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(script_content[match.start() : idx + 1])
                    break
    return calls


def validate_generated_script(
    script_content: str,
    review_agents: Optional[list[dict]] = None,
) -> tuple[bool, list[str]]:
    """Fail closed on malformed, inert, unbounded, or unsafe workflow source."""
    del review_agents  # Review evidence is enforced by the engine, never inferred here.
    if not script_content or not script_content.strip():
        return False, ["Script content is empty"]

    errors: list[str] = []
    first = script_content.lstrip()
    if not re.match(r'''^(export|/[/*]|const |let |var |"use strict"|'use strict')''', first):
        errors.append("Script starts with non-JavaScript text")

    required_patterns = (
        (r"export\s+const\s+meta\s*=", "Missing `export const meta =` declaration"),
        (r'''\bname\s*:\s*["'`]''', "Meta object missing `name` field"),
        (r'''\bdescription\s*:\s*["'`]''', "Meta object missing `description` field"),
        (r"export\s+default\s+(async\s+)?function", "Missing `export default function`"),
    )
    for pattern, message in required_patterns:
        if not re.search(pattern, script_content):
            errors.append(message)

    executable = _strip_js_strings_and_comments(script_content)
    calls_agent = bool(re.search(r"\bagent\s*\(", executable))
    calls_pattern = bool(
        re.search(r"\b(classify|fanout|verify|generate|tournament|loop|sequence|race)\s*\(", executable)
    )
    if not (calls_agent or calls_pattern):
        errors.append("No `agent()` or 6+2 orchestration primitive call found")
    if re.search(r"\bworkflow\s*\(", executable):
        errors.append("`workflow()` sub-workflow references are no longer supported")
    if re.search(r"\b(parallel|sequence|race)\s*\(\s*\[\s*\]\s*\)", executable):
        errors.append("Empty orchestration arrays do not dispatch work")

    agent_calls = _iter_js_call_sources(script_content, "agent")
    labels = [
        match.group(2)
        for call in agent_calls
        for match in re.finditer(r'''\blabel\s*:\s*(["'])(.*?)\1''', call)
    ]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        errors.append("Duplicate agent label(s): " + ", ".join(duplicates))
    for index, call in enumerate(agent_calls, 1):
        if not re.search(r"\btimeout\s*:", call):
            match = re.search(r'''\blabel\s*:\s*(["'])(.*?)\1''', call)
            errors.append(f"Direct agent call `{match.group(2) if match else index}` lacks `timeout:`")
    if agent_calls and not re.search(r"(\.\s*error\b|\bcatch\s*\(|\btry\s*\{)", executable):
        errors.append("Direct agent calls must handle `result.error` or use try/catch")

    delimiters = (("{", "}", "braces"), ("[", "]", "brackets"), ("(", ")", "parentheses"))
    for opening, closing, name in delimiters:
        balance = executable.count(opening) - executable.count(closing)
        if balance:
            errors.append(f"Unbalanced {name}: {balance:+d}")

    for pattern, description in _DANGEROUS_PATTERNS:
        if re.search(pattern, script_content):
            errors.append(f"[capability] Forbidden pattern: {description}")

    if errors:
        logger.warning("Script validation failed with %d error(s): %s", len(errors), "; ".join(errors))
    return not errors, errors


def _matching_brace(source: str, start: int) -> int | None:
    masked = _strip_js_strings_and_comments(source)
    depth = 0
    for index in range(start, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_meta_from_script(script_content: str) -> Optional[dict[str, Any]]:
    """Extract the static JSON-like `meta` object; dynamic metadata fails closed."""
    match = re.search(r"export\s+const\s+meta\s*=\s*(\{)", script_content or "")
    if not match:
        return None
    end = _matching_brace(script_content, match.start(1))
    if end is None:
        return None
    try:
        meta = json.loads(_js_object_to_json(script_content[match.start(1) : end + 1]))
    except json.JSONDecodeError as exc:
        logger.debug("Failed to parse workflow meta: %r", exc)
        return None
    return meta if isinstance(meta, dict) else None


def _js_object_to_json(js_object: str) -> str:
    """Normalize the intentionally small static meta literal subset."""
    result = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", js_object)
    result = _replace_js_quote(result, "'")
    result = _replace_js_quote(result, "`")
    result = re.sub(
        r'(?<=[{,\n])\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:',
        r' "\1":',
        result,
    )
    return re.sub(r",\s*([}\]])", r"\1", result)


def _replace_js_quote(source: str, quote: str) -> str:
    """Convert one JS string delimiter to JSON quotes without evaluating source."""
    output: list[str] = []
    index = 0
    while index < len(source):
        current = source[index]
        if current in ("'", '"', "`") and current != quote:
            delimiter = current
            output.append(current)
            index += 1
            while index < len(source):
                char = source[index]
                output.append(char)
                index += 1
                if char == "\\" and index < len(source):
                    output.append(source[index])
                    index += 1
                elif char == delimiter:
                    break
            continue
        if current != quote:
            output.append(current)
            index += 1
            continue
        index += 1
        value: list[str] = []
        while index < len(source):
            char = source[index]
            if char == "\\" and index + 1 < len(source):
                nxt = source[index + 1]
                value.append(nxt if nxt == quote else char + nxt)
                index += 2
            elif char == quote:
                index += 1
                break
            else:
                value.append(char)
                index += 1
        output.append(json.dumps("".join(value), ensure_ascii=False))
    return "".join(output)


def generate_simple_script(
    requirement: str,
    selected_tools: list[str] | None = None,
    tool_model_map: dict[str, str] | None = None,
) -> str:
    """Return the one-Agent automatic path used by lightweight callers/tests."""
    tools = [tool for tool in (selected_tools or ["coco"]) if tool] or ["coco"]
    primary = tools[0]
    model = (tool_model_map or {}).get(primary)
    prompt = f"""Fulfill the user's request exactly and self-check the result before returning.

Requirement:
{requirement}

If the user asks for analysis only, planning only, or says not to change code, do not change code.
For implementation, return the complete production result. On a blocker, return a clear error instead of asking the user.

{get_subagent_encouragement()}""".strip()

    return f'''export const meta = {{
  name: "automatic-focused-workflow",
  description: "One focused Agent call with bounded automatic failure handling",
  phases: [{{ title: "Execution", detail: "Complete and self-check the requested task" }}],
  maxConcurrent: 1,
  tools: {json.dumps(tools)},
  patterns: [],
}};

export default async function main() {{
  phase("Execution");
  log("Executing one focused task");
  const result = await agent({{
    prompt: {json.dumps(prompt, ensure_ascii=False)},
    tool: {json.dumps(primary)},
    model: {json.dumps(model)},
    role: "focused-executor",
    label: "execute-focused-task",
    timeout: 180,
  }});
  if (result && result.error) {{
    return {{ error: result.error, fallback: true, stage: "Execution", partial: null }};
  }}
  return completionEnvelope(result);
}}

function completionEnvelope(result) {{
  const data = result && typeof result === "object" ? result : {{}};
  return {{
    card_summary: {{
      verdict: data.error ? "needs_attention" : "passed",
      conclusion: data.summary || data.conclusion || "任务已完成，完整结果见报告。",
      findings: [],
      verification: [{{ status: "passed", text: "执行 Agent 已完成自检。" }}],
      deliverables: Array.isArray(data.deliverables) ? data.deliverables : [],
      next_steps: Array.isArray(data.next_steps) ? data.next_steps : [],
    }},
    result,
    verification: null,
  }};
}}
'''
