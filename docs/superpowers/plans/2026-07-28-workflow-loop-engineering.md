# Workflow Loop Engineering Adaptation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 借鉴 Loop Engineering 文档中的确定性编排、结构化节点契约和 pipeline-first 原则，降低 GhostAP `/wf` 的突发并发、无效重试与错误结果扩散，同时保留现有 Dynamic Workflow 原语和资源模型。

**Architecture:** Python `AgentExecutor` 继续作为结构化输出重试的唯一权威层；Node runtime 只做一次防御性校验并把失配转成显式错误，避免两层重试相乘。`pipeline()` 保持“item 间并行、item 内 stage 串行、结果顺序稳定”的现有语义，但复用 `parallel()` 的有界调度器，不再一次性启动全部 item。脚本生成提示改为先侦察/定范围，再按任务依赖选择 pipeline、parallel barrier 或高阶模式。

**Tech Stack:** Python 3.11、Node.js VM runtime、JSON-RPC、pytest、uv。

## Adaptation Boundaries

- 保留 GhostAP 的 `DEFAULT_MAX_CONCURRENT=10`、`HARD_MAX_CONCURRENT=16`、`MAX_TOTAL_AGENTS=200`；不照搬文档中的 1000 Agent 规模。
- 保留现有独立调用 Journal key 与 Dynamic Workflow 八类原语；本轮不引入新的 chained resume 协议。
- 结构化输出沿用 GhostAP 的紧凑 shape schema（字符串类型名和 JSON 示例值），不新增第三方 JSON Schema 依赖。
- 不改变 `/wf` 三步选 Agent/自动执行交互，也不修改卡片 UI。

---

### Task 1: 为 pipeline 有界调度补回归契约

**Files:**
- Modify: `tests/test_workflow_parallel_perf.py`

**Interfaces:**
- Consumes: runtime init 的 `max_concurrent`。
- Produces: pipeline item 并发不超过上限、仍保持并行/map 和输入顺序的测试契约。

- [x] **Step 1: 修改源码语义测试**

将“pipeline 必须直接使用 `Promise.all`”改为“pipeline 必须复用有界 `parallel()` 调度”，同时继续断言文档说明 item 间并行、stage 间串行。

- [x] **Step 2: 添加 Node 集成测试**

构造 4 个 pipeline item、`max_concurrent=2`，断言耗时接近两批而非一批或四批，并确认结果数量/顺序不变。

- [x] **Step 3: 运行测试确认红灯**

Run: `uv run python -m pytest tests/test_workflow_parallel_perf.py::TestPipelineSemantics -q`

Expected: 旧 runtime 因 pipeline 仍直接 `Promise.all(items.map(...))` 而失败。

### Task 2: 实现 pipeline 有界并行

**Files:**
- Modify: `src/workflow_engine/runtime/runtime.js`

**Interfaces:**
- Consumes: `parallel()`、`maxConcurrent`、pipeline stages/options。
- Produces: 有界 item 调度、稳定结果顺序、原有 fail-fast/continueOnFailure/abort 行为。

- [x] **Step 1: 复用 parallel 调度**

把每个 pipeline item 包装为一个异步 task，交给 `parallel()`；每个 task 内仍逐 stage `await`。

- [x] **Step 2: 保留失败语义**

`continueOnFailure=true` 返回结构化局部失败；默认首错时中止未启动任务并 abort 已在执行的 Agent 请求。

- [x] **Step 3: 运行定向测试确认绿灯**

Run: `uv run python -m pytest tests/test_workflow_parallel_perf.py::TestPipelineSemantics tests/test_workflow_fault_tolerance.py::TestPipelineErrorHandling -q`

Expected: 全部通过。

### Task 3: 为结构化输出契约补失败测试

**Files:**
- Modify: `tests/test_workflow_parallel_perf.py`
- Modify: `tests/test_workflow_fault_tolerance.py`

**Interfaces:**
- Consumes: compact shape schema，例如 `{"summary": "", "findings": [], "done": false}`。
- Produces: 类型/嵌套结构校验、重试耗尽显式错误、单层重试权威契约。

- [x] **Step 1: 添加类型与嵌套结构测试**

覆盖错误数组类型、布尔/数字示例值、嵌套对象和单元素数组 item schema。

- [x] **Step 2: 收紧重试耗尽契约**

修改既有测试：保留最后一次原始输出用于诊断，但 `AgentCallResult.error` 必须明确标记结构化输出失败。

- [x] **Step 3: 添加 JS 单次调用测试**

Host 返回不匹配 schema 时，JS 不得再次发起完整 `agent_call`；应返回 `{error}`。

- [x] **Step 4: 运行测试确认红灯**

Run: `uv run python -m pytest tests/test_workflow_parallel_perf.py::TestSchemaValidation tests/test_workflow_fault_tolerance.py::TestSchemaValidationRetry -q`

Expected: 旧实现因只检查 key、耗尽后仍无 error 而失败。

### Task 4: 实现类型化 shape schema 与单层重试

**Files:**
- Modify: `src/workflow_engine/executor.py`
- Modify: `src/workflow_engine/runtime/runtime.js`

**Interfaces:**
- Consumes: schema type names、JSON exemplar values、AgentCallResult。
- Produces: Python 权威校验/重试，JS 防御性校验，无 3×3 重试放大。

- [x] **Step 1: 实现递归 shape 校验**

支持 `string/array/object/number/integer/boolean/null/any` 及常用别名；`[]` 表示任意数组，`[schema]` 校验元素，嵌套 dict 要求对应 key 与类型。

- [x] **Step 2: 重试耗尽返回显式错误**

Python 保留 `output`，清空 `parsed`，并设置稳定、可诊断的 schema validation error。

- [x] **Step 3: JS 取消外层整次重试**

优先传播 host `result.error`；host 返回数据后只做一次同语义校验，失配直接返回 error，不再重复 `agent_call`。

- [x] **Step 4: 运行定向测试确认绿灯**

Run: `uv run python -m pytest tests/test_workflow_parallel_perf.py::TestSchemaValidation tests/test_workflow_fault_tolerance.py::TestSchemaValidationRetry -q`

Expected: 全部通过。

### Task 5: 调整工作流生成策略

**Files:**
- Modify: `src/workflow_engine/script_gen.py`
- Modify: `tests/test_workflow_parallel_perf.py`
- Modify: `tests/test_workflow_api_contract.py`

**Interfaces:**
- Consumes: 用户需求、现有 primitives 和资源上限。
- Produces: scope-first、pipeline-first、barrier 按需、成本与失败边界明确的生成提示。

- [x] **Step 1: 添加提示契约测试**

断言 prompt 明确要求：先发现/限定 worklist；同构多阶段任务优先 pipeline；只有下游需要全部结果时才使用 barrier；结构化 schema 是节点契约。

- [x] **Step 2: 更新 prompt**

修正文档与示例，说明 pipeline 是有界并行 map；补充范围上限和结构化输出错误处理；避免“最大化并行”诱导无界 fan-out。

- [x] **Step 3: 运行定向测试**

Run: `uv run python -m pytest tests/test_workflow_parallel_perf.py::TestPipelineSemantics tests/test_workflow_api_contract.py -q`

Expected: 全部通过。

### Task 6: 扩大验证并记录项目决策

**Files:**
- Modify: `.Memory/2026-07-28.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: 本次 diff、红绿测试与文档对照结论。
- Produces: 可追溯的实现、原因、验证、未采纳项和后续风险。

- [x] **Step 1: 运行 Workflow 相关测试**

Run: `uv run python -m pytest tests/test_workflow_parallel_perf.py tests/test_workflow_fault_tolerance.py tests/test_workflow_runtime_primitives.py tests/test_workflow_api_contract.py tests/test_workflow_confirm.py -q -m "not slow"`

Expected: 全部通过。

- [x] **Step 2: 运行 Workflow 扩展测试**

Run: `uv run python -m pytest tests/ -q -m "not slow" -k workflow`

Expected: 全部通过。

- [x] **Step 3: 运行静态与配置检查**

Run: `uv run ruff check src/workflow_engine/executor.py src/workflow_engine/script_gen.py tests/test_workflow_parallel_perf.py tests/test_workflow_fault_tolerance.py`

Run: `node --check src/workflow_engine/runtime/runtime.js`

Run: `uv run python -m src.main --validate`

Run: `git diff --check`

Expected: 全部通过且无错误。

- [x] **Step 4: 更新 Memory 并复核原始范围**

记录借鉴项、GhostAP 适配边界、性能收益和剩余风险；确认稳定性、性能和错误可见性均有回归证据。
