# GhostAP Dynamic Multi-Agent Orchestration V3 Implementation Plan

> 历史说明（2026-08-20）：本文中的 deny-all、风险分级和工具过滤方案不再是当前执行契约；GhostAP 已将执行权限交给后端自身处理。

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 GhostAP 建设成后端中立、角色动态、拓扑动态、证据驱动且可耐久恢复的多 Agent 协作控制面，使用户选定的本机 Agent 能按任务需要自动分工、执行、验证、重规划和交付，而不把 Codex、Grok、TraeX、Hermes、OpenClaw 等品牌写死为固定角色。

**Architecture:** 保留 Direct、Deep、Spec 的既有执行语义，以 Workflow 作为首个 Orchestration V3 迁移入口。新增确定性的 Orchestration Kernel、类型化 PlanGraph、独立 Journal/Blob 存储和后端中立 DispatchPort；临时 ACP Agent 与持久 Employee 仅共享资源和派发合约，不共享身份、会话、长期记忆或权限实例。

**Tech Stack:** Python 3.13、Pydantic frozen models、ACP、Node.js Workflow compatibility runtime、pytest、`uv`、本地 fsync/flock Journal、加密 BlobStore、飞书/Lark SDK。

## Global Constraints

- 本文是架构与迁移实施计划，不包含生产代码改动。
- 核心 planner、matcher、scheduler、verifier 和 topology policy 不得按 backend、model 或厂商品牌分支。
- 用户选择的是允许使用的资源池；默认不强制池中每个 Agent 都被调用。
- Direct 保持原 Agent、原会话和零额外规划模型跳数。
- Deep、Spec 保持各自受保护的算法、完成判定和恢复协议，不改写为通用 Workflow。
- Workflow 保留当前 owner-bound Agent Pool 单次确认契约；确认后全自动执行，不增加脚本确认、继续批准或手动恢复门。
- 未来只有在用户显式保存项目默认 Squad 后，才能省略每次任务的池确认；这是独立产品变更。
- Agent Pool 确认不等于文件写入、删除、部署、凭据使用或外部副作用授权。
- Planner 使用 deny-all 会话，只能提出类型化计划修订，不能调用工具、发送消息或直接派发 Agent。
- 未建立独立工作区前，所有会修改共享工作树、git index 或外部对象的节点必须串行。
- `UNKNOWN`、缺失证据、无有效 verifier、未解决 Effect 或非权威子代理状态均不得计为成功。
- 所有节点数、调用数、并发数、讨论轮数、层级深度、重规划次数、token、墙钟和成本预算必须有界。
- 新旧 Run 不得在运行中切换执行引擎；旧版本 Run 由原执行器恢复到终态。
- 仅使用 `uv`，不得使用 pip 或 conda。
- 所有行为改动遵循 TDD：先观察目标回归测试失败，再写最小实现，再运行扩大测试。
- 每个阶段形成独立、聚焦的 commit；仅在用户明确要求时 push。

---

## 1. Executive Decision

GhostAP 的定位不是大模型底座，而是本机 Agent 的安全协作控制面：

```text
用户定义目标、资源边界和授权
             ↓
GhostAP 规范化任务、选择拓扑、分配资源、控制预算和权限
             ↓
不同 Agent 负责语义工作、产出 Artifact 和 Evidence
             ↓
GhostAP 根据验收标准验证、重规划并决定唯一终态
```

以下设计均被明确否决：

- “Codex 永远负责编排、Grok 永远评审、TraeX 永远修改”之类的品牌角色映射。
- 对所有任务强制执行“设计 → 编码 → 评审 → 修改”的固定流水线。
- 用户选择多个 Agent 后，无论任务复杂度都强制逐个调用。
- 允许 Agent 自由群聊、自由拉起新 Agent、自由扩大池和预算。
- 将所有临时 ACP Agent 包装成 Employee，或让所有模式进入 Employee Gateway。
- 用模型输出的非空文本直接认定任务完成。
- 在共享 cwd 中允许多个 Agent 并行修改代码。

目标模型是：**动态角色 + 动态任务图 + 确定性控制内核 + 后端中立适配器。**

## 2. Current Implementation Assessment

### 2.1 可直接保留的基础

1. `src/workflow_engine/agent_pool.py`
   - 已有稳定 `A1`、`A2` 等运行内 Agent ID。
   - 已冻结 `agent_id -> tool/model/profile/effort` 绑定。
   - 已拒绝完全相同的重复绑定。

2. `src/workflow_engine/run_spec.py`
   - 已能将运行限制在冻结池内。
   - host 是 tool/model 的权威来源，生成脚本不能覆盖真实绑定。

3. `src/workflow_engine/runtime/runtime.js`
   - 已有 `classify`、`fanout`、`verify`、`generate`、`tournament`、`loop`、`sequence`、`race` 等通用拓扑原语。
   - 这些原语可以继续作为兼容执行机制或 PlanGraph 编译目标。

4. `src/autonomous/journal/` 与 `src/autonomous/journal/blob_store.py`
   - 已有哈希链、CAS、flock、fsync、锚定和加密 Blob 等耐久组件。
   - 应复用实现和不变量，但 Orchestration 使用独立日志实例或命名空间。

5. `src/autonomous/runtime/employee_actor.py`
   - 持久 Employee 的串行邮箱、warm session、身份和长期上下文语义正确。
   - 应作为一种持久资源 Adapter 保留，而不是成为所有 Agent 的统一生命周期。

6. `src/trust/action_matrix.py` 与 ACP permission checks
   - 默认拒绝和一次性安全授权边界必须完整保留。

### 2.2 当前阻碍目标的具体问题

1. 品牌仍进入核心选择逻辑。
   - `src/workflow_engine/script_gen.py` 的 `_CAPABILITY_NOTES` 按品牌描述能力。
   - `src/workflow_engine/agent_pool.py` 与 `src/feishu/handlers/workflow.py` 有固定推荐顺序。
   - `src/workflow_engine/constants.py` 存在具体默认编排器。
   - `runtime.js` 的个别 fallback 仍引用具体 backend。

2. Workflow 的动态候选尚未成为真实运行时选择。
   - `candidateAgentIds` 主要用于元数据、展示和静态校验。
   - 实际 `agent()` 调用仍需要单一 `agentId`。
   - 系统缺少针对 WorkUnit 的 runtime capability matcher 和 dispatch 前复核。

3. Employee Team 仍是固定阶段。
   - `src/autonomous/team/models.py` 定义固定 phase。
   - `src/autonomous/team/coordinator.py` 实际执行 execute → review → optional revise。
   - 多个执行者仍可能共享同一 role/instruction，不能表达独立子任务 DAG。

4. Workflow 缺少权威结构化数据流。
   - Agent 输出主要作为字符串传递。
   - fanout 综合前会截断成员输出。
   - 当前 Workflow journal 更接近结果缓存，不是耐久控制日志。

5. 并行写缺少隔离。
   - Workflow 可以并发调用多个 Agent，但默认共享同一 cwd。
   - 外层 RepoLock 无法隔离同一 Workflow 内部的并行写。

6. 当前预算只是近似调用次数。
   - 没有完整覆盖 planner、fallback、schema repair、retry、reviewer、provider 内部子代理、token、墙钟和金额。

7. ACP Provider 目录不具备可信调度能力模型。
   - 当前主要描述名称、启动命令、可用性和模型归一化。
   - 尚未声明 cancel/resume、tool filtering、sandbox、结构化输出、workspace、成本遥测等能力。

## 3. Product Semantics

### 3.1 Resource Pool 语义

Pool member 的含义是 `ADMITTED`：本次运行被允许使用的资源，而不是必须调用的固定岗位。

对每个 WorkUnit，Kernel 基于精确目录 revision 派生：

```text
ADMITTED → AVAILABLE → ELIGIBLE → ASSIGNED → CONTRIBUTED → VERIFIED
```

- `ADMITTED`：进入本次冻结池。
- `AVAILABLE`：当前健康、未退役、租户和项目边界有效。
- `ELIGIBLE`：满足当前 WorkUnit 的硬能力、权限、上下文和 deadline 条件。
- `ASSIGNED`：Kernel 为具体 attempt 预留预算后正式派发。
- `CONTRIBUTED`：产生权威 Artifact、Evidence 或结构化决策。
- `VERIFIED`：其产出已通过对应验收标准。

`eligible` 不是 Agent 的永久属性，而是 `(agent_id, work_unit_id, directory_revision)` 的派生关系。

### 3.2 参与策略

每次 Run 明确一种参与策略：

| Policy | 含义 | 适用情况 |
| --- | --- | --- |
| `adaptive` | Kernel 只选择真正需要的资源子集 | 默认策略；简单任务允许单 Agent |
| `all_contribute` | 每个 admitted resource 必须至少产生一份符合 schema 的贡献 | 用户明确要求全员观点或竞赛 |
| `pinned` | 指定 Agent 必须参与指定 criterion 或 WorkUnit | 合规、身份、上下文负责人等明确约束 |

池中未被调用的 Agent 不算失败；UI 必须区分“已选择”和“实际贡献”。

### 3.3 全自动契约

- Pool 冻结后，计划、分工、Agent 讨论、执行、验证、修订和终态化全部自动推进。
- 普通格式错误、Agent 提问、暂时失败和 review 不确定使用有界自动恢复。
- 只有缺少用户独有信息、权限或外部前提时进入 `BLOCKED` 明确终态。
- 用户补充信息时创建 successor Run 或新 PlanVersion，不静默复活已终态 Run。
- 高风险操作未获原始授权时拒绝危险部分，继续可安全完成的部分，并如实报告 coverage gap。

## 4. Target Architecture

```text
Feishu / future API / CLI
          │
          ▼
Ingress: identity + provenance + trust + dedupe + project context
          │
          ▼
AdmissionDecision
  ├── Direct Lane
  │     └── existing session, zero extra planner hop
  ├── Protected Lane
  │     ├── Deep
  │     └── Spec
  └── Collaborative Orchestration V3
        │
        ├── TaskIntent normalization
        ├── ResourcePool freeze
        ├── deny-all semantic Planner → typed PlanRevision
        ├── deterministic Validator / Kernel / Scheduler
        ├── dedicated OrchestrationStore
        ├── DispatchPort
        │     ├── EphemeralACPAdapter
        │     ├── EmployeeDispatchAdapter
        │     └── future backend adapters
        ├── ArtifactStore + EvidenceStore
        ├── deterministic and semantic verification
        ├── bounded replanning
        └── Finalization + Feishu RunSnapshot
```

### 4.1 Planner 与 Kernel 的边界

Planner 负责语义工作：

- 解析目标和验收标准。
- 提议 WorkUnit、依赖关系、RoleBrief 和拓扑。
- 在失败证据出现后提议 PlanDelta。

Planner 不具备以下权力：

- 不能调用工具或写文件。
- 不能直接 dispatch Agent。
- 不能扩展 ResourcePool。
- 不能扩大权限、预算或副作用范围。
- 不能认定 Run 成功。

Kernel 负责确定性控制：

- 校验 schema、DAG、节点和循环上限。
- 执行硬能力和权限过滤。
- dispatch 前重新验证 TOCTOU 状态。
- 预留和结算预算。
- 控制并发、取消、迟到结果和重试。
- 根据 Evidence 和未决 Effect 决定唯一终态。

### 4.2 Agent 之间的协作协议

不开放无约束自由群聊。所有协作消息使用 `CollaborationEnvelope`，至少包含：

- `run_id`
- `plan_version`
- `work_unit_id`
- `sender_agent_id`
- `recipient_agent_id`
- `purpose`
- `in_reply_to`
- `payload_schema`
- `artifact_refs`
- `budget_slice`
- `deadline`

允许的语义动作包括：

- `proposal`
- `evidence`
- `critique`
- `revision`
- `decision`
- `handoff_request`

只有 Kernel 可以接受 handoff、创建新节点或分配新预算。

## 5. Canonical Domain Contracts

### 5.1 TaskIntent

`TaskIntent` 是版本化、冻结的任务事实，包含：

- `intent_id`、`version`、`source_request_id`
- objectives 与 deliverables
- HARD / SOFT acceptance criteria
- user constraints 与 system constraints，并记录来源
- mutation scope 与 risk class
- authorization scope
- context references
- budget 与 hard deadline
- output contract
- assumptions 与 unresolved facts

### 5.2 AgentResource 与 ExecutionBinding

`AgentResource` 只描述逻辑资源：

- stable resource ID
- `EPHEMERAL` 或 `PERSISTENT`
- capability assertions
- authority grants
- health、load、reliability、latency、cost class
- context affinity
- failure domain

`ExecutionBinding` 隔离物理实现：

- opaque binding ID
- adapter ID
- model selection
- transport configuration
- workspace binding
- session capabilities

backend 品牌只允许出现在 ExecutionBinding、adapter 注册、审计和 UI 标签中。

### 5.3 CapabilityAssertion 与 CapabilityRequirement

能力使用 namespaced ID，例如：

- `artifact.source.read`
- `artifact.source.write`
- `execution.shell`
- `execution.test`
- `analysis.security`
- `evidence.web.primary_source`
- `modality.vision`
- `session.cancel`
- `session.resume`
- `transport.structured_output`

Assertion 至少记录：

- source
- revision
- confidence
- observed_at
- expires_at
- supporting evidence

Requirement 使用 `HARD` 或 `PREFERRED`，要求的是能力，不是品牌。

### 5.4 RoleBrief

RoleBrief 属于 WorkUnit，不属于 Agent，包含：

- objective
- perspective
- input artifact references
- output schema
- forbidden actions
- rubric
- independence constraints
- evidence requirements

RoleBrief 不授予权限。

### 5.5 PlanGraph 与 WorkUnit

每个 `PlanGraph` 都是不可变版本对象：

- `plan_id`
- `version`
- `supersedes`
- `trigger`
- `directory_revision`
- `work_units`
- typed dependency edges
- bounded topology groups
- global budget
- plan hash

WorkUnit 至少包含：

- objective 与 RoleBrief
- input artifact refs
- output schema
- required/preferred capabilities
- acceptance criteria
- side-effect class
- artifact ownership
- independence constraints
- retry/replan policy
- node budget 与 deadline

Dependency edge 类型：

- `DATA`
- `CONTROL`
- `REVIEW_OF`
- `COMPARES`
- `INVALIDATES`

单个 PlanVersion 必须是 DAG。循环、辩论轮次和递归委派通过有界 TopologyGroup 或新 PlanVersion 表达，不允许任意图环。

### 5.6 NodeResult、Artifact 与 Evidence

节点间不得再以截断字符串作为权威数据。`NodeResult` 至少包含：

- full output BlobRef
- parsed payload
- artifact refs
- evidence refs
- protocol stop reason
- token、cost、wall-clock 与 attempt count
- effect summary
- authority summary
- input digest
- repository/workspace revision digest
- producer attempt ID 与 provenance

每个 Artifact 必须有单写者，或者由显式 merge/integrator 节点产生。

### 5.7 BudgetEnvelope

预算必须覆盖：

- logical WorkUnit 数量
- Agent attempt 数量
- planner 调用
- fallback、retry 和 schema repair
- verifier 与 reviser
- provider 内部可观测子代理
- token
- 配置化成本等级或金额
- wall-clock deadline
- concurrent slots
- replan rounds
- hierarchy depth
- debate/tournament rounds

Kernel 必须在 dispatch 前预留预算，attempt 终态后结算；预算不足时不得先调用再记账。

## 6. Dynamic Topology Policy

| Topology | 选择条件 | 禁止条件 |
| --- | --- | --- |
| Single Agent | 单一目标、上下文连续、一个主要产物、有确定性 oracle | 不能为了展示多 Agent 而拆分 |
| Sequence | 存在数据依赖、共享可变状态或外部副作用 | 不得伪装成并行 |
| Fan-out + Synthesize | 子问题独立、输入可复制、输出有明确 merge owner | 共享 cwd 写入或没有综合标准 |
| Generate + Tournament | 需要多个替代方案且有统一、可复验 rubric | 无合法 judge、无隔离或 contender 自评 |
| Structured Debate | 高影响歧义、互斥解释、暂无确定性 oracle | 普通代码问题、轮数不受限、自由拉人 |
| Race | 首个有效结果足够且 loser 可安全取消 | 最快不等于最好或副作用不可取消 |
| Hierarchical Delegation | 大型跨域任务、子图边界清晰、上下文明显超载 | 简单任务、无深度和预算上限 |

拓扑选择依据任务依赖、风险、证据要求和工作区冲突，不依据 backend 名称。

## 7. Verification and Replanning

### 7.1 验证顺序

1. Schema、类型、policy、digest 等本地确定性检查。
2. 单元测试、集成测试、静态分析等可执行 oracle。
3. 只有无法确定性判断的 criterion 才调用语义 verifier。
4. 高风险 criterion 根据明确 independence constraint 选择 verifier。

不同厂商不自动等于独立。独立性应基于 Agent ID、session lineage、上下文来源和 failure domain 的审计证明。

每条 acceptance criterion 得到独立 verdict：

- `PASS`
- `FAIL`
- `UNKNOWN`
- `CONFLICT`

只有所有 HARD criterion 为 PASS、最终 Artifact 已耐久化且所有 Effect 已处置，Run 才能成功。

### 7.2 重规划触发

- 没有 eligible Agent。
- dispatch 前能力、权限或健康状态变化。
- execution、schema 或工具协议失败。
- verifier 返回 FAIL、UNKNOWN 或 CONFLICT。
- deadline 或预算风险超过阈值。
- 外部依赖或输入 revision 改变。

每次重规划创建 `PlanGraph vN+1`：

- 不原地覆盖旧计划。
- 保留输入 digest 未变化的成功 Artifact。
- 输入变化的后代节点标记 INVALIDATED。
- committed side effect 不得重放，除非具有相同 idempotency key 或显式 compensation。
- 使用 plan hash + failure fingerprint 防止振荡。
- 重规划上限耗尽后进入明确终态。

## 8. Run Lifecycle and Terminal Semantics

```text
ADMITTED
  → INTENT_RESOLVING
  → PLANNING
  → EXECUTING
  ↔ VERIFYING
  ↔ REPLANNING
  → FINALIZING
  → terminal
```

这些是生命周期阶段，不是固定的 Agent 角色流水线。

终态定义：

- `SUCCEEDED`：全部 HARD criterion 有 PASS 证据，最终 Artifact durable，Effect 全部处置。
- `FAILED`：执行、验证或重规划有界耗尽，或出现不可恢复错误。
- `TIMED_OUT`：hard deadline 到期，所有 in-flight attempt 已取消或处置。
- `CANCELLED`：授权用户主动停止，取消和 Effect 清理完成。
- `BLOCKED`：缺少不可自动获得的权限、输入或外部前提；这是明确终态，不是无限等待态。

partial Artifact 可以随 FAILED/BLOCKED 一起交付，但不得使用 `PARTIAL_SUCCESS` 模糊成功语义。

## 9. Storage and Recovery

### 9.1 独立 OrchestrationStore

复用现有 JournalWriter、Anchor 和 BlobStore 实现，但使用独立实例、目录和事件命名空间，避免与 Employee 全局 Journal 共用吞吐和故障域。

建议权威事件族：

- `orchestration.run.admitted`
- `orchestration.pool.frozen`
- `orchestration.intent.resolved`
- `orchestration.plan.revised`
- `orchestration.node.ready`
- `orchestration.attempt.prepared`
- `orchestration.attempt.executing`
- `orchestration.attempt.terminal`
- `orchestration.artifact.committed`
- `orchestration.verification.recorded`
- `orchestration.replan.requested`
- `orchestration.effect.prepared`
- `orchestration.effect.committed`
- `orchestration.effect.disposed`
- `orchestration.run.finalization_started`
- `orchestration.run.terminal`

流式 token、短期 UI 动画和重复进度不进入 canonical Journal，只进入可丢失的 progress cache。

### 9.2 恢复等级

第一阶段只承诺：

- 已耐久 plan、artifact、evidence 和 terminal 可以确定性重放。
- 纯只读、幂等且输入快照一致的节点可以自动重试。
- 未知在途副作用明确进入失败或阻塞，不宣称自动续跑成功。

只有在补齐 idempotency key、effect reconciler 和可重建数据边后，才扩大自动恢复范围。

## 10. Workspace and Mutation Safety

### MVP

- 所有 read-only 节点允许受限并行。
- 所有 mutating 节点全局串行。
- Planner、reviewer 和只读分析节点使用 deny-all 或只读 tool policy。

### 后续隔离模型

- 每个 mutating attempt 使用独立 git worktree 或等价隔离工作区。
- Agent 交付 patch/commit Artifact，不直接修改主工作区。
- 单一 Integrator 节点负责合并。
- 合并前检查 base revision、冲突、测试和权限。
- merge 失败进入新 PlanVersion，不允许多个 Agent 直接竞争 git index。

这是一种内部安全原语，不重新引入已退役的用户可见 Worktree 产品模式。

## 11. Module Boundaries

### 11.1 新模块

| File | Responsibility |
| --- | --- |
| `src/orchestration/contracts.py` | TaskIntent、ResourcePool、RoleBrief、PlanGraph、Attempt、Result、Budget、Authority 等冻结合约 |
| `src/orchestration/catalog.py` | Capability catalog、hard filter、soft ranking、eligibility reason codes |
| `src/orchestration/planning.py` | deny-all planner、PlanDelta 解析和 PlanRevision 生成 |
| `src/orchestration/validation.py` | schema、DAG、pool、权限、预算、拓扑和 workspace 冲突校验 |
| `src/orchestration/projection.py` | Journal 事件纯 reducer 和 RunSnapshot |
| `src/orchestration/store.py` | 独立 Journal/Blob adapter 和序列化版本 |
| `src/orchestration/scheduler.py` | ready-node 调度、预算预留、attempt 生命周期、取消和重规划 |
| `src/orchestration/artifacts.py` | Artifact/Evidence 写入、digest、provenance 和引用解析 |
| `src/orchestration/ports.py` | PlannerPort、DispatchPort、VerifierPort、Clock 和 Store protocols |
| `src/orchestration/adapters/ephemeral_acp.py` | 运行级一次性 ACP 会话适配 |
| `src/orchestration/adapters/employee.py` | 持久 Employee Gateway/Actor 适配，不泄漏员工身份语义 |
| `src/agent_session/catalog.py` | BackendDescriptor、SessionCapabilities、SessionRequest |

### 11.2 现有模块处置

| Existing module | Plan |
| --- | --- |
| `src/workflow_engine/agent_pool.py` | 保留冻结 ID/binding 不变量，逐步适配 ResourcePool |
| `src/workflow_engine/run_spec.py` | 保留 legacy reader 与 replay adapter；V3 新写入改用共享 contracts |
| `src/workflow_engine/runtime/runtime.js` | 暂作兼容执行器和语义对照，不再作为权威计划 |
| `src/workflow_engine/script_gen.py` | 先移除品牌说明，再由 typed planner 取代 JS/正则权威计划 |
| `src/workflow_engine/journal.py` | 明确为 result cache，不能冒充 canonical Journal |
| `src/autonomous/team/coordinator.py` | 旧 Team Run 继续使用；新 Run 迁移后停止固定 phase 写入 |
| `src/autonomous/gateway/` | 保持 Employee 专属执行和信任边界，通过 adapter 接入 |
| `src/feishu/handlers/workflow.py` | 最终只负责 admission、资源池 UI、停止和 RunSnapshot 展示 |
| `src/feishu/ws_client.py` | 保留身份、来源、信任、去重和队列边界，不承载编排领域状态 |

## 12. Phased Implementation Plan

### Task 0: Contract Guardrails and Shadow Mapping

**Files:**

- Create: `src/orchestration/__init__.py`
- Create: `src/orchestration/contracts.py`
- Create: `src/orchestration/legacy_mapping.py`
- Create: `tests/orchestration/test_contracts.py`
- Create: `tests/orchestration/test_legacy_mapping.py`
- Create: `tests/orchestration/test_brand_neutrality.py`
- Test: `tests/contracts/test_direct_programming_lane.py`
- Test: `tests/contracts/test_protected_execution_lanes.py`
- Test: `tests/test_workflow_agent_pool_contract.py`

**Interfaces:**

- Consumes: existing `WorkflowRunSpec`, `AgentPool`, Team v2 snapshots.
- Produces: frozen, versioned `TaskIntent`, `ResourceBinding`, `ResourcePool`, `RoleBrief`, `WorkUnit`, `PlanGraph`, `NodeResult`, `BudgetEnvelope`, `AuthorityEnvelope`.

- [ ] 写序列化 round-trip、未知字段拒绝、版本不匹配拒绝和 frozen mutation 回归。
- [ ] 写 brand rename metamorphic test：仅替换 adapter/model 标签，逻辑计划输入保持一致。
- [ ] 写 legacy Workflow/Team 只读映射测试，禁止 shadow path dispatch。
- [ ] 运行 `uv run python -m pytest tests/orchestration/test_contracts.py tests/orchestration/test_legacy_mapping.py tests/orchestration/test_brand_neutrality.py -q`，确认新合约尚不存在时 RED。
- [ ] 实现最小 immutable contracts 和只读 mapping，不修改生产路由。
- [ ] 运行上述测试并确认 PASS。
- [ ] 运行 Direct、Protected Lane 和 Agent Pool 合同测试，确认既有产品语义未变化。
- [ ] 创建聚焦 commit：`feat(orchestration): add versioned neutral contracts`。

**Rollback:** 删除 shadow mapping 入口；现有 Run、Journal 和路由没有状态变化。

### Task 1: Backend Catalog and SessionRequest

**Files:**

- Create: `src/agent_session/catalog.py`
- Modify: `src/acp/provider.py`
- Modify: `src/acp/providers/__init__.py`
- Modify: `src/agent_session/factory.py`
- Modify: `src/agent_session/backend_resolver.py`
- Modify: `src/workflow_engine/tool_registry.py`
- Test: `tests/test_agent_session.py`
- Test: `tests/test_acp_protocol_contract.py`
- Test: `tests/test_workflow_tool_registry.py`
- Create: `tests/orchestration/test_backend_catalog.py`

**Interfaces:**

- Consumes: existing provider registrations and session factory arguments.
- Produces: `BackendDescriptor`, `SessionCapabilities`, `SessionRequest`, adapter conformance result and health revision.

- [ ] 写现有七类 provider descriptor 兼容测试。
- [ ] 写未知 provider 在 Orchestration Lane 中默认不可调度的失败关闭测试。
- [ ] 写 fake Hermes/OpenClaw descriptor 测试，证明无需修改 planner、handler 或 topology policy 即可进入目录。
- [ ] 写 cancel、resume、structured output、tool filtering、workspace 和 cost telemetry 能力缺失 reason-code 测试。
- [ ] 运行目标测试并观察缺少 catalog 时 RED。
- [ ] 用兼容 adapter 包装现有 provider；旧 `create_engine_session()` 继续转译到 SessionRequest。
- [ ] 从核心 matcher 和 prompt 中删除品牌能力推断、固定推荐顺序和具体 default orchestrator。
- [ ] 运行 ACP、Agent Session、Workflow Tool Registry 回归并确认 PASS。
- [ ] 创建聚焦 commit：`refactor(agent-session): add backend capability catalog`。

**Rollback:** 兼容入口直接调用旧 factory；Orchestration feature flag 保持关闭。

### Task 2: Orchestration Journal, Blob and Projection

**Files:**

- Create: `src/orchestration/store.py`
- Create: `src/orchestration/events.py`
- Create: `src/orchestration/projection.py`
- Create: `tests/orchestration/test_projection.py`
- Create: `tests/orchestration/test_store_recovery.py`
- Test: `tests/autonomous/unit/test_journal_writer.py`

**Interfaces:**

- Consumes: existing JournalWriter、Anchor、BlobStore implementations.
- Produces: isolated `OrchestrationStore.append()`, deterministic `reduce_event()`, immutable `RunSnapshot`.

- [ ] 写每种 canonical event 的 schema 和 reducer transition 测试。
- [ ] 写截断 frame、anchor CAS 失败、Blob 缺失、重复 event 和 sequence gap 测试。
- [ ] 写同一 Journal 重放得到相同 plan/attempt/effect/terminal 的确定性测试。
- [ ] 写 terminal 单调和 unresolved Effect 禁止终态测试。
- [ ] 运行 store/projection 测试并观察缺少实现时 RED。
- [ ] 建立独立目录、事件命名空间和 Blob 引用；不写入 Employee 全局 Journal。
- [ ] 确认 progress/token event 不进入 canonical Journal。
- [ ] 运行 Orchestration store 与现有 Journal writer 回归并确认 PASS。
- [ ] 创建聚焦 commit：`feat(orchestration): add durable run store and projection`。

**Rollback:** 停止 V3 admission 并保留 reader；不删除任何已写 Journal。

### Task 3: Typed Planner, Validator and Shadow Plan

**Files:**

- Create: `src/orchestration/planning.py`
- Create: `src/orchestration/validation.py`
- Create: `src/orchestration/catalog.py`
- Create: `tests/orchestration/test_planner_contract.py`
- Create: `tests/orchestration/test_plan_validation.py`
- Create: `tests/orchestration/test_topology_policy.py`
- Modify: `src/config/settings.py`

**Interfaces:**

- Consumes: `TaskIntent`, frozen `ResourcePool`, `AgentDirectorySnapshot`, `BudgetEnvelope`.
- Produces: schema-validated `PlanDelta`, immutable `PlanRevision`, `EligibilityDecision` with stable reason codes.

- [ ] 写 planner deny-all 与 pool-bound 测试，确认不能扩池、改 binding、扩大权限或调用工具。
- [ ] 写 DAG、边类型、fanout、loop、hierarchy、node count、concurrency 和 deadline 上限测试。
- [ ] 写 hard-filter 与 soft-ranking 测试；权限、租户、stale health、输入不可达必须先排除。
- [ ] 写 simple-task proportionality 测试，确认默认只生成一个 execution node。
- [ ] 写 same-resource/different-RoleBrief 测试。
- [ ] 运行目标测试并观察缺少 planner/validator 时 RED。
- [ ] 接入 shadow plan：对现有 Workflow 生成类型化计划但绝不 dispatch。
- [ ] 记录 v1 JS metadata 与 typed PlanRevision 的可比较差异，不改变用户结果。
- [ ] 运行目标测试和 Workflow generation/security 回归并确认 PASS。
- [ ] 创建聚焦 commit：`feat(orchestration): add typed shadow planner`。

**Rollback:** 关闭 shadow-plan 设置；不影响 v1 Workflow 生成或执行。

### Task 4: Workflow V3 Execution MVP

**Files:**

- Create: `src/orchestration/ports.py`
- Create: `src/orchestration/scheduler.py`
- Create: `src/orchestration/artifacts.py`
- Create: `src/orchestration/adapters/__init__.py`
- Create: `src/orchestration/adapters/ephemeral_acp.py`
- Modify: `src/feishu/handlers/workflow.py`
- Modify: `src/workflow_engine/executor.py`
- Modify: `src/workflow_engine/manager.py`
- Create: `tests/orchestration/test_scheduler.py`
- Create: `tests/orchestration/test_ephemeral_acp_adapter.py`
- Create: `tests/orchestration/test_workflow_v3_integration.py`
- Test: `tests/test_workflow_execution_bindings.py`
- Test: `tests/test_workflow_auto_execute.py`
- Test: `tests/test_workflow_executor_cancel.py`

**Interfaces:**

- Consumes: validated PlanRevision、ResourcePool、OrchestrationStore、SessionRequest.
- Produces: `AttemptSpec`, `AttemptOutcome`, structured `NodeResult`, Artifact/Evidence refs and RunSnapshot.

- [ ] 写只有 ready node 可派发、预算先预留、dispatch 前重新验证 eligibility 的测试。
- [ ] 写池外 agentId、script tool/model 覆盖、迟到 attempt 和重复 terminal 的失败关闭测试。
- [ ] 写 read-only 节点可并行、mutating 节点必须串行的调度测试。
- [ ] 写结构化 NodeResult 测试，禁止截断字符串成为权威输入。
- [ ] 写用户停止、hard deadline 和 executor cancellation 测试。
- [ ] 运行目标测试并观察 V3 scheduler/adapter 缺失时 RED。
- [ ] 仅对 feature-flag 命中的新 Workflow Run 使用 V3；旧 Run 保持 v1 owner。
- [ ] Pool 单次确认后自动完成 typed planning、execution 和 terminalization，不增加中间确认。
- [ ] 未知 in-flight 副作用进入明确失败，不声称自动恢复。
- [ ] 运行 Workflow pool、binding、security、cancel、lifecycle 和 V3 integration 测试并确认 PASS。
- [ ] 创建聚焦 commit：`feat(workflow): execute new runs through orchestration v3`。

**Rollback:** 仅关闭新 Run admission；已进入 V3 的 Run 继续由 V3 恢复到终态。

### Task 5: Verification, Replanning and Full Budgeting

**Files:**

- Create: `src/orchestration/verification.py`
- Create: `src/orchestration/replanning.py`
- Create: `src/orchestration/budget.py`
- Modify: `src/orchestration/scheduler.py`
- Create: `tests/orchestration/test_verification.py`
- Create: `tests/orchestration/test_replanning.py`
- Create: `tests/orchestration/test_budget.py`
- Test: `tests/test_workflow_fault_tolerance.py`
- Test: `tests/test_workflow_runtime_primitives.py`

**Interfaces:**

- Consumes: criterion definitions、NodeResult、Evidence、current PlanVersion、remaining budget and Effect ledger.
- Produces: criterion-level verdict、VerificationDecision、ReplanRequest、PlanGraph vN+1 and budget settlements.

- [ ] 写 deterministic-oracle-first 测试，已有可靠测试结果时不调用语义 verifier。
- [ ] 写零有效 verifier、UNKNOWN、CONFLICT、证据缺失和 independence 不满足均不能成功的测试。
- [ ] 写 FAIL 后生成新 PlanVersion、保留有效 Artifact、失效后代节点的测试。
- [ ] 写 committed side effect 不重复、相同 failure fingerprint 不振荡的测试。
- [ ] 写 planner、retry、fallback、schema repair、reviewer、token、deadline 和 replan 都计入预算的测试。
- [ ] 写预算无法预留时不发起外部调用的测试。
- [ ] 运行目标测试并观察 verification/replanning/budget 缺失时 RED。
- [ ] 实现有界验证和重规划闭环，只有 Kernel 能认定 terminal。
- [ ] 运行 Orchestration 与 Workflow fault-tolerance/runtime regressions 并确认 PASS。
- [ ] 创建聚焦 commit：`feat(orchestration): add evidence gates and bounded replanning`。

**Rollback:** 关闭 V3 新 admission；已有 V3 Run 仍由相同版本 reducer 和 scheduler 收尾。

### Task 6: Isolated Mutation and Integrator

**Files:**

- Create: `src/orchestration/workspaces.py`
- Create: `src/orchestration/integration.py`
- Modify: `src/orchestration/scheduler.py`
- Create: `tests/orchestration/test_workspace_isolation.py`
- Create: `tests/orchestration/test_patch_integration.py`
- Test: `tests/test_workflow_ac4_isolation.py`

**Interfaces:**

- Consumes: base repository revision、mutating WorkUnit、workspace authority、patch/commit Artifact.
- Produces: isolated workspace lease、candidate patch Artifact、Integrator merge result and conflict Evidence.

- [ ] 写两个 mutating node 绝不共享工作目录或 git index 的测试。
- [ ] 写 base revision 变化、冲突 patch、越权文件和 symlink escape 的拒绝测试。
- [ ] 写只有 Integrator 能合并到目标工作区的测试。
- [ ] 写 merge 后测试失败触发 replan 而不是直接成功的测试。
- [ ] 运行 isolation tests 并观察缺少内部工作区服务时 RED。
- [ ] 引入内部 worktree/patch Artifact 和单一 Integrator。
- [ ] 仅在隔离、权限和 merge owner 全部成立时允许并行 mutation。
- [ ] 运行 isolation、security、workflow integration 回归并确认 PASS。
- [ ] 创建聚焦 commit：`feat(orchestration): isolate parallel mutations`。

**Rollback:** 配置退回 mutating-node 全局串行；已经生成的 patch Artifact 保留审计，不自动合并。

### Task 7: Persistent Employee Adapter

**Files:**

- Create: `src/orchestration/adapters/employee.py`
- Modify: `src/autonomous/team/service.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/autonomous/gateway/coordinator.py`
- Create: `tests/orchestration/test_employee_adapter.py`
- Create: `tests/orchestration/test_mixed_resource_pool.py`
- Test: `tests/autonomous/integration/test_employee_team_gateway.py`
- Test: `tests/autonomous/integration/test_employee_runtime_recovery.py`

**Interfaces:**

- Consumes: `AttemptSpec`、persistent Employee resource ID、existing Employee dispatch binding.
- Produces: neutral `AttemptOutcome` and Artifact/Evidence refs without exposing Employee private memory or credentials.

- [ ] 写 Employee 和 ephemeral ACP 可以同时进入 ResourcePool 的测试。
- [ ] 写临时 Agent 不能读取、写入或冒用 Employee 长期记忆、身份、Channel 和权限的测试。
- [ ] 写 Employee context affinity 参与软排序但不形成固定 planner 角色的测试。
- [ ] 写 V3 attempt 与 Employee Journal/Actor 状态因果绑定和唯一终态测试。
- [ ] 运行 adapter 与 Employee gateway/recovery 测试并观察缺少 adapter 时 RED。
- [ ] 通过 DispatchPort 接入现有 Employee Gateway/Actor，不重写 Employee 生命周期。
- [ ] 仅让新 Team admission 使用 V3 feature flag；Team v2 active Run 继续由旧 coordinator 完成。
- [ ] 运行 mixed pool、Employee integration、recovery 和 tool-isolation 回归并确认 PASS。
- [ ] 创建聚焦 commit：`feat(orchestration): add persistent employee adapter`。

**Rollback:** 新 Team admission 切回 v2；已进入 V3 的 Team Run 不切换 owner。

### Task 8: Feishu Admission, Saved Squads and Truthful UI

**Files:**

- Create: `src/orchestration/squads.py`
- Modify: `src/feishu/route_decision.py`
- Modify: `src/feishu/handlers/workflow.py`
- Modify: `src/workflow_engine/renderer.py`
- Modify: `src/feishu/ws_client.py`
- Create: `tests/orchestration/test_saved_squads.py`
- Test: `tests/test_workflow_agent_selection_contract.py`
- Test: `tests/test_workflow_renderer.py`
- Test: `tests/test_workflow_progress_summary.py`
- Test: `tests/test_ws_client_routing.py`

**Interfaces:**

- Consumes: owner-bound Squad revision、TaskIntent、RunSnapshot.
- Produces: frozen ResourcePool、AdmissionDecision and truthful Feishu card projection.

- [ ] 写默认 Squad 只能由 owner 创建、修改和绑定项目的测试。
- [ ] 写每个 task 冻结精确 Squad revision、后续修改不影响在途 Run 的测试。
- [ ] 写无默认 Squad 时继续使用当前一次池确认的兼容测试。
- [ ] 写卡片区分 admitted、eligible、assigned、contributed、verified 的测试。
- [ ] 写未调用的 admitted Agent 不显示为失败、非权威嵌套子代理不驱动终态的测试。
- [ ] 写确认后无脚本确认、继续批准和手动恢复门的测试。
- [ ] 运行 selection、renderer、progress 和 routing 测试并观察新语义缺失时 RED。
- [ ] 将飞书 Handler 收敛为 admission 和 RunSnapshot 展示，不再拥有执行真相。
- [ ] 保留 trust、provenance、dedupe、SMART shell 和 protected lane 路由边界。
- [ ] 运行飞书与 Orchestration integration 回归并确认 PASS。
- [ ] 创建聚焦 commit：`feat(feishu): add versioned squads and run snapshots`。

**Rollback:** 关闭 saved Squad 自动准入，恢复每个 Workflow Run 的一次池确认；不改变已冻结 Run。

### Task 9: Legacy Drain and Retirement

**Files:**

- Modify: `src/workflow_engine/manager.py`
- Modify: `src/workflow_engine/script_gen.py`
- Modify: `src/workflow_engine/runtime/runtime.js`
- Modify: `src/autonomous/team/coordinator.py`
- Modify: `src/autonomous/team/projection.py`
- Modify: `src/workflow_engine/journal.py`
- Create: `scripts/audit_orchestration_run_versions.py`
- Create: `tests/orchestration/test_legacy_readers.py`
- Create: `tests/orchestration/test_run_version_audit.py`

**Interfaces:**

- Consumes: v1 Workflow、v2 Team、v3 Orchestration persisted records.
- Produces: active-run version audit、legacy read-only projection and safe retirement evidence.

- [ ] 写 v1/v2/v3 历史记录均可读取、但只有各自 owner 可以写终态的测试。
- [ ] 写 active v1/v2 非零时禁止删除旧 writer/executor 的测试。
- [ ] 写 migration audit 脚本，输出各版本非终态 Run 和无法解析记录。
- [ ] 先停止旧版本新 admission，保留 legacy reader 至少一个发布周期。
- [ ] 确认 active v1/v2 为零、历史状态和附件可读取后，再删除固定 Team 写路径。
- [ ] typed PlanGraph 覆盖原语、取消、预算、恢复和报告后，再移除 JS 作为权威计划。
- [ ] 将 `WorkflowJournal` 更名为真实的 ResultCache，避免与 canonical Journal 混淆。
- [ ] 运行 legacy reader、version audit、Workflow 和 Autonomous regression suites。
- [ ] 创建聚焦 commit：`refactor(orchestration): retire drained legacy writers`。

**Rollback:** 删除前以聚焦 commit 为恢复点；绝不通过重新解释历史事件进行回滚。

### Task 10: Final Integration and Release Evidence

**Files:**

- Modify: `docs/product-contract.md`
- Modify: `docs/testing.md`
- Modify: `docs/acp_provider_guide.md`
- Modify: `.Memory/{YYYY-MM-DD}.md`
- Modify: `.Memory/Abstract.md`

- [ ] 回读本计划的全部目标、红线和验收范围，建立 requirement-to-test 对照表。
- [ ] 运行 `uv run python -m pytest tests/orchestration/ -q`，期望全部 PASS 且无 skip 隐藏核心路径。
- [ ] 运行 `uv run python -m pytest tests/test_workflow_agent_pool_contract.py tests/test_workflow_execution_bindings.py tests/test_workflow_security.py tests/test_workflow_fault_tolerance.py tests/test_workflow_runtime_primitives.py -q`，期望全部 PASS。
- [ ] 运行 `uv run python -m pytest tests/contracts/test_direct_programming_lane.py tests/contracts/test_protected_execution_lanes.py tests/test_ws_client_routing.py -q`，期望全部 PASS。
- [ ] 运行 `uv run python -m pytest tests/autonomous/ -q`，期望 Employee Journal、Gateway、Actor 和 Team recovery 全部 PASS。
- [ ] 运行 `uv run ruff check src tests`，期望 0 error。
- [ ] 运行 `uv run python -m src.main --validate`，期望配置和生产组装验证通过。
- [ ] 运行 `uv run python scripts/test_inventory.py tests/`，期望测试资产检查通过。
- [ ] 运行 `git diff --check`，期望无空白错误。
- [ ] 执行 brand scan，确认具体 backend 名只存在于 adapter、配置、UI、兼容 fixture 和文档示例，不存在于核心 planner/matcher/topology/verifier。
- [ ] 执行 crash injection：planning、prepared、executing、verification、replanning、finalization 各边界恢复结果确定且无重复 Effect。
- [ ] 执行真实飞书租户验收：一次池确认或已保存 Squad 后全自动完成，卡片参与状态与权威 Journal 一致。
- [ ] 更新产品契约、测试指南、ACP provider 接入指南和 Memory 证据。
- [ ] 创建聚焦 commit：`docs(orchestration): record v3 release contracts and evidence`。

## 13. Acceptance Matrix

| Requirement | Required evidence |
| --- | --- |
| 后端中立 | backend rename metamorphic tests；核心 brand scan 为零 |
| 动态角色 | 同一资源在不同 WorkUnit 使用不同 RoleBrief |
| 动态参与 | adaptive 可只用一个 Agent；all_contribute 和 pinned 可显式强制参与 |
| 动态拓扑 | single/sequence/fanout/tournament/debate/race/hierarchy 均有选择与边界测试 |
| Pool 不可扩展 | planner、script、child coordinator 和 Agent 均无法引入池外资源 |
| 权限不扩大 | dispatch 权限是 user/project/resource/node 交集，TOCTOU 变化失败关闭 |
| 证据终态 | HARD criteria 全 PASS、Artifact durable、Effect 已处置才成功 |
| 自动重规划 | 新 PlanVersion、Artifact 复用和失效、无振荡、无副作用重复 |
| 并行安全 | MVP 写节点串行；隔离上线后 patch + Integrator 才能并行写 |
| 耐久恢复 | 各 crash boundary 可重放；未知副作用不伪装成成功 |
| 成本有界 | 所有显式和隐藏调用均预留、结算并受总预算限制 |
| Employee 隔离 | 临时 ACP Agent 不共享 Employee 身份、Channel、长期记忆和权限 |
| 既有通道不回归 | Direct 零额外 hop；Deep/Spec 算法不变；SMART shell 仍走 shell |
| 飞书全自动 | Pool 冻结后无中间确认；卡片如实展示实际参与和证据 |
| 新后端可插拔 | fake Hermes/OpenClaw 只注册 descriptor/adapter 即可进入池 |

## 14. Recommended First Deliverable

第一批可交付范围只做到 Task 0–4：

- 建立后端中立合约和 Backend Catalog。
- 建立独立 Orchestration Journal/Projection。
- typed Planner 先影子运行。
- 仅让 feature-flag 下的新 Workflow Run 进入 V3。
- 动态角色和拓扑真实生效。
- read-only 可并行，所有写节点串行。
- 使用结构化 NodeResult 和 Evidence 决定终态。
- 未知副作用崩溃明确失败，不虚构自动恢复。

这一范围已经能够证明 GhostAP 的核心方向成立：任意合规 Agent 可通过统一资源合约进入池，系统按任务动态分工，而不是按品牌执行固定流水线。同时它不要求一次性重写 Employee、Direct、Deep、Spec 或立即解决并行代码合并的全部问题。

达到 Task 0–4 的发布门后，再依次投入完整预算与重规划、隔离并行写、Employee Adapter 和 saved Squad，风险最低。
