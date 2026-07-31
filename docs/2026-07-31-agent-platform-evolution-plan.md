# GhostAP Agent Platform Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development`（推荐，同一会话并行）或 `executing-plans`（独立会话）逐任务实施；每个行为变更使用 `test-driven-development`。执行时把任务内的编号步骤复制为可勾选工作清单。

**Goal:** 在不牺牲普通 Agent 直连编程、Deep 与 Spec 成熟路径的前提下，把 GhostAP 演进为一个飞书原生的 Agent Department 平台：既能稳定连接多种 Agent，也能以可恢复、可审计、可授权的方式编排 Agent，并允许 Agent 在明确边界内主动工作。

**Architecture:** 采用“一个控制面、三条执行通道、一个耐久安全内核”。普通编程保持零额外 LLM 跳数的 Direct Lane；Deep/Spec 作为 Protected Strategy Lane 保持现有实现和交互；Worktree/Workflow/Team/Slock 作为 Development Lane，优先承接统一 Agent 能力目录、类型化编排 IR、耐久调度和主动触发的演进。三条通道共享只读任务视图、控制协议、策略门、审计事件和 Agent 后端能力目录，但不强制共享内部执行算法。

**Tech Stack:** Python 3.11+、`uv`、Pydantic、frozen dataclass、现有 ACP/CLI 传输、Node.js Workflow DSL、Feishu `lark-oapi` / `lark-channel-sdk`、现役 Journal/Blob/Vault 与待收敛的 production Policy/Dispatch ports、pytest、ruff。

**Status:** 仅为实施计划；本轮未授权生产代码、配置、运行时或卡片行为变更。

**Deployment assumption:** 当前只有 Owner 自己在一台受信主机上使用 GhostAP。功能通过对应的自动化合同、安全和恢复测试后，立即在 Owner 的正常菜单或命令中可见、可体验；本计划不建设按租户灰度、Beta/RC/production 晋级、长观察窗口、发布 allowlist 或签名发布证据包。

---

## 中文执行摘要

### 单用户直接体验约束

这条约束覆盖既有计划中的发布节奏设计，但不降低工程正确性和安全边界：

- **完成即开放：** 一个功能完成相关自动化合同/故障测试和最小 Owner smoke 后，直接对当前 Owner 开放，不再等待灰度、放量、版本晋级或人工设备验收。
- **标签只表达完成度：** `mature`、`developing`、`not_implemented` 只帮助 Owner 理解能力成熟度，不控制菜单曝光，也不形成发布状态机。
- **开关只用于止损：** 每个新子系统最多保留一个紧急关闭/回退开关；开关不能演化成租户分组、百分比放量或多级发布配置。
- **开放后继续验真：** 桌面/移动端卡片、长时间运行和更高容量属于开放后的体验/诊断清单；发现问题直接修复或紧急回退，不反向建设发布流程。
- **短时 dry-run 仍保留：** IR compiler、迁移 comparator、恢复演练和离线 replay 是工程验证步骤，不是线上观察期；一旦对应检查通过，不再人为延迟 Owner 使用。
- **单用户不等于无权限：** 主 Bot 可能收到群成员、外部事件和 Agent 生成内容。Owner 身份、项目/聊天范围、数据外发、预算、审批、Effect 锚定、kill、备份和恢复仍是硬边界。
- **内部命名空间可以保留：** `tenant_key` 可作为当前部署的固定安全/恢复命名空间，防止事件、Artifact 和授权串线；它不再是发布或功能开放维度。

### 核心判断

GhostAP 当前不是“缺少 Agent 能力”，而是已经拥有多套较强能力，却尚未形成一个边界清晰、可以持续扩展的 Agent 平台：

- 普通编程已经能直接连接 Coco、Claude、Aiden、Codex、Gemini、Trae/TTADK 等后端，并维护项目与会话状态。
- Deep、Spec 已形成相对成熟的专项执行路径。
- Worktree、Workflow 已具备隔离执行和动态编排能力，但仍处于开发、收敛和恢复语义补齐阶段。
- Employee Runtime 已具备独立飞书 Bot 身份、加密凭据、持久 Actor、复用会话、上下文和恢复能力。
- Team Coordinator、Slock、Workflow 分别覆盖了部分多 Agent 协作问题，但任务、状态、编排和恢复模型仍有重叠。
- Autonomous v5 已经提供 Journal、Effect、Policy/Dispatch 原型和 Vault 等高价值安全基座；其中 Journal/Vault 已进入 Employee production composition，部分旧 DispatchGate/Policy/Manager/Scheduler 仍未接入或不满足最终锚定合同，不能直接视为现役能力。

因此，下一阶段最重要的不是继续增加新的顶层模式，也不是把成熟引擎全部重写成一个“大一统工作流”，而是完成以下四个闭环：

1. **Agent 接入闭环**：增加一个 Agent 后端时，只需要声明能力、实现一个驱动并通过一致性测试，不再修改多个引擎中的 `if agent_type == ...`。
2. **编排闭环**：把开发中的 Workflow、Team、Slock 和 Worktree 接到同一个可恢复任务/产物/Effect 模型上，同时保留它们各自的产品职责。
3. **主动工作闭环**：由耐久 Trigger 产生有权限边界、有预算、有截止时间、有去重键的 Goal/Run，而不是让 Agent 无限制轮询或“自己决定一直干活”。
4. **产品闭环**：用户看到的是项目、员工、团队、任务、审批和审计；Agent 后端、模型和执行策略仍然可直接选择，但不再成为彼此冲突的顶层产品分类。

### 必须保护的成熟路径

本计划把以下内容视为不可回归的产品合同，而不是待统一重写的遗留能力：

| 通道 | 当前定位 | 本计划允许的近期变化 | 本计划禁止的近期变化 |
| --- | --- | --- | --- |
| 普通 Agent 编程 | 核心、成熟、最高频 | 统一后端描述、兼容工厂、只读任务投影、观测 | 增加规划 Agent、强制生成编排图、增加确认步骤、改变会话续接语义 |
| Deep | 成熟专项策略 | 暴露统一状态/停止/摘要适配器，修复已证实缺陷 | 为统一 IR 重写内部算法或改变用户入口 |
| Spec | 成熟专项策略 | 暴露统一状态/停止/摘要适配器，保持 provider-local retry | 为统一 IR 重写阶段模型或跨 provider 静默回退 |
| Worktree | 开发中隔离策略 | 作为受控节点接入任务/Effect/产物合同 | 把它泛化成所有任务的默认执行方式 |
| Workflow | 开发中编排产品 | 作为类型化编排 IR 的首个试点，保留 JS 兼容前端；通过对应测试后立即供 Owner 使用 | 一次性替换现有运行时或绕过正确性测试开启真实 Effect |
| Team | 开发中多员工执行 | 成为耐久多员工编排执行核心候选 | 与 Slock 各自维护第二套任务事实源 |
| Slock | 开发中群协作入口 | 保留分类、选择性唤醒、群交互和兼容入口 | 继续扩张为另一套独立执行内核 |

普通编程的硬性验收标准是：**用户明确选择某个 Agent 时，系统不得额外调用任何规划/路由 LLM；不得因为统一控制面而增加一次远程往返；原会话键、取消、重试、流式卡片和项目状态语义保持不变。**

### 推荐的产品北极星

> GhostAP 是一个飞书/Lark 原生的 Agent Department Gateway。人可以直接与一个 Agent 编程，也可以雇佣有独立身份和长期上下文的员工，把复杂目标交给团队或工作流，并在审批、预算、审计和恢复边界内让这些 Agent 主动工作。

这一定义同时容纳三种用户心智：

- “我现在就要和 Codex 直接写代码。”
- “这个复杂问题交给 Deep/Spec，或者让一组 Agent 编排完成。”
- “持续关注这个目标，条件满足后主动处理，只在例外或高风险动作时找我。”

### 与既有计划的关系

本计划不是对 [`docs/2026-07-30-ghostap-product-convergence-plan.md`](./2026-07-30-ghostap-product-convergence-plan.md) 的重复。该计划解决的是产品合同、访问控制、统一上下文/路由/任务控制、运行生命周期和恢复等基础收敛问题；其中 Task 1–24 仍应被视为本计划的 **Foundation Track**。原 Task 25 的签名租户 Beta Gate 不再是前置，Task 26 只保留本机切换、回退和验证，不做分批发布。

本计划在其上新增三类内容：

- 如何把“支持多个工具”升级为真正可扩展的 Agent 接入平台。
- 如何在保护成熟通道的条件下形成耐久的多 Agent 编排核心。
- 如何把 Trigger、Goal、Policy、Scheduler、Team 和 Employee Actor 接成主动工作闭环。

若两个计划发生冲突，以本计划的“成熟路径保护红线”“单用户直接体验约束”和后文工程检查点为准；任何 Deep/Spec/普通编程内部改造都必须单独证明不会改变现有合同。

---

## 1. 分析依据与事实优先级

### 1.1 事实来源

本计划基于以下仓库内事实，而不是仅依据 README 的产品描述：

1. 生产代码和测试。
2. `.Memory/Abstract.md` 与 2026-07 的逐日实施记录。
3. `docs/goals.md`、Agent Department 设计、Employee Runtime ADR 和现有收敛计划。
4. README、帮助文本和示例。

当这些来源不一致时，优先采用代码与测试。当前 README 对 Autonomous 生产接线的部分描述已经落后于 `EmployeeDepartmentRuntime` 的实际实现，因此文档差异本身也是待治理的产品问题。

### 1.2 重点阅读入口

- Agent 会话：`src/agent_session/`、`src/acp/`、`src/ttadk/`
- 普通编程入口：`src/feishu/dispatcher.py`、`src/feishu/handlers/`
- 成熟策略：`src/deep_engine/`、`src/spec_engine/`
- 开发中策略：`src/worktree_engine/`、`src/workflow_engine/`
- 群协作：`src/slock_engine/`
- Employee/Team/主动系统：`src/autonomous/`
- 卡片与任务展示：`src/card/`
- 本地决策：`.Memory/`

### 1.3 术语必须拆开

当前代码中的 `agent_type` 在不同位置可能同时代表工具、provider、传输方式、员工身份或执行角色。后续设计必须使用以下独立概念：

| 概念 | 含义 | 示例 |
| --- | --- | --- |
| Employee | 飞书中的长期 Agent 身份，拥有凭据、记忆、工作区和审计历史 | “代码审查员小林” |
| Backend | 实际执行推理/编程工作的工具或 provider | Codex、Coco、Claude |
| Transport | 如何启动并交换消息 | ACP、CLI bridge、未来的远程协议 |
| Model | Backend 内部选择的模型 | provider-specific model id |
| Capability | 后端经过验证能够完成的合同 | streaming、resume、image input、cancel |
| Role | 某次任务中的职责 | planner、coder、reviewer |
| Session | Backend 的一次可续接执行上下文 | ACP/CLI session |
| Strategy | 如何组织一次工作 | direct、deep、spec、worktree、workflow、team |
| Skill | 可注入提示或工具的领域能力 | 测试、代码审查、飞书操作 |

只有拆开这些概念，才能做到“同一个员工可以换 Backend”“同一个 Backend 可以承担不同角色”“编排器根据真实能力选 Agent”，并避免继续增加后端特定分支。

---

## 2. 当前实现能力矩阵

### 2.1 总体成熟度

| 能力域 | 已有实现与证据 | 当前成熟度 | 主要缺口 |
| --- | --- | --- | --- |
| 普通 Agent 直连 | `agent_session`、ACP/CLI adapter、项目/聊天模式、主卡流式输出 | **成熟，需保护** | 后端分支分散、工厂重复、能力声明不统一 |
| Deep | 独立 manager、持久化 hook、卡片与取消路径 | **成熟，需保护** | 与统一任务控制仅有弱适配，恢复证据仍可增强 |
| Spec | 完整阶段/评审/持久化体系、provider-local 逻辑 | **成熟，需保护** | 生命周期与统一控制面尚未标准化 |
| Worktree | 隔离工作树、合并/报告路径 | **开发中** | timeout/cancel/终态聚合及作为编排节点的合同 |
| Workflow | JS DSL、11 个模板、8 类高阶原语、选择/确认/进度 UI | **开发中但能力丰富** | 当前 Journal 更接近结果缓存；运行拓扑、Effect、恢复不是耐久事实 |
| Employee Runtime | Hire、独立 Bot、Vault、Channel 子进程、Actor、上下文、恢复 | **单机运行架构已成形** | 当前 Owner 环境的端到端验证、运维可见性和故障恢复体验 |
| Team v2 | `TeamCoordinatorActor`、冻结模型、投影、加密 Blob、边界限制 | **开发中/试运行** | 有限阶段机，不是通用 DAG；与 Slock/Workflow 职责重叠 |
| Slock | 分类、自动解析、activation guard、任务队列、群协作上下文 | **开发中/兼容层** | 与 Employee Team 的事实源、调度和任务模型重复 |
| Autonomous 安全内核 | 现役 Journal/Vault/frozen domain，加上 Effect/Policy/Dispatch 原型 | **架构强项但尚未全线接通** | 生产 Employee 使用独立 dispatch coordinator；旧 Gate/Policy/Manager/Scheduler 不可直接当作生产事实 |
| 主动 Trigger | Trigger/Subscription/Scheduler 类型和局部实现 | **未形成生产闭环** | cron 求值、投影恢复、去重、misfire、生产 composition、权限包络 |
| 产品控制面 | 飞书命令、卡片、主 Bot、员工 Bot | **功能多但表面分散** | 项目/员工/团队/任务/审批视图未完全成为统一产品入口 |
| 本机体验验证 | 大量单元/集成/chaos/acceptance 测试 | **自动化证据强** | 需补 Owner 飞书端到端、桌面/移动端、重启与备份恢复清单 |

### 2.2 Agent 接入层的真实状态

已有的 `src/agent_session/protocol.py::SyncSession` 提供了有价值的最小会话协议：启动、载入、发送、重试、取消、关闭、快照和健康检查。`src/acp/provider.py::ACPProvider` 也提供了 provider 可用性、启动命令和模型归一化等抽象。

但“增加一个 Agent”仍然不是插件式操作：

- `src/agent_session/factory.py` 和 `src/acp/session_factory.py` 存在两套创建路径。
- Claude CLI、TTADK CLI、ACP provider 和模型归一化在工厂及多个引擎中重复分支。
- `src/agent_session/backend_resolver.py` 主要把后端归类为 `acp|cli`，不足以表达图片、流式、续接、取消、工具过滤、上下文长度、并发和信任能力。
- `src/workflow_engine/tool_registry.py` 又建立了一套工具描述、缓存和 fallback。
- 能力是否可用主要依靠命名规则和启动探测，而不是一组可执行的一致性测试。

因此当前状态可以准确描述为：**支持多种 Agent，但尚未达到“低成本、可验证地接入任意新 Agent”的平台化水平。**

### 2.3 编排层的真实状态

Workflow Runtime 已支持 `agent`、`parallel`、`pipeline`、`phase`、`workflow`，以及 `classify`、`fanout`、`verify`、`generate`、`tournament`、`loop`、`sequence`、`race` 等高阶原语，并具备超时、取消、并发上限、循环上限和 host function 包装。

它的优势是表达力高、生成灵活、产品交互已经存在；局限是：

- JS 脚本同时承担计划、动态控制和执行描述，难以在执行前得到完整、稳定的任务图。
- `src/workflow_engine/journal.py::WorkflowJournal` 的职责更接近按 prompt/tool/model/role/schema 缓存结果，不等价于 Autonomous Journal 的生命周期与 Effect 事实源。
- 节点输入输出大多是文本或松散 JSON，缺少跨 Agent 的类型化 Artifact 合同。
- Workflow、Team、Slock、Worktree 各自拥有一部分状态、取消、重试和完成语义。
- 计划中途重规划、未知外部 Effect、恢复后重新派发等行为还没有统一规则。

Team v2 已有更强的耐久 Actor 与投影，但它采用有界的 planning → dispatch → review → revise → finalize 阶段机。当前代码明确限制每次 team run 的 turn、assignment、fanout 和 handoff；这对安全非常有价值，但它不是可表达任意 DAG 的通用编排内核。

### 2.4 主动工作层的真实状态

生产 `EmployeeDepartmentRuntime` 已经组合了 Journal、Vault、Channel、Ingress、Router、Outbox、Employee Actor、Team Coordinator 和恢复。它提供了构建主动系统最难得的基础：长期身份、耐久输入、可恢复会话和安全投递。

与此同时：

- `src/autonomous/scheduler/triggers.py` 仍以进程内可变映射为主，`schedule_cron` 与 `next_fire_at` 尚未构成完整的耐久 cron 语义。
- `src/autonomous/domain/goals.py` 已有 `TriggerSubscription`，但“订阅 → occurrence → admission → plan → run → notification”的生产组合路径仍不完整。
- 较早的 `src/autonomous/coordinator.py`、`bootstrap.py`、manager-only 配置与新的 Employee production composition 并存。
- Policy/standing order 的概念已有实现，但尚未成为所有主动 Effect 的统一生产门。

所以当前可以说“有自主系统的内核和大量零件”，不能说“Agent 已经可以在生产中按计划或事件长期主动完成目标”。

---

## 3. 根因，而不是表面功能缺口

### 根因 A：身份、后端、模型、角色和策略被同一个字段或同一层承担

直接后果是每增加一个 Backend 或 Transport，都可能触碰多个引擎、选择卡、诊断和恢复逻辑。编排器也无法基于真实能力进行选择。

### 根因 B：多套执行策略缺少共同的最小控制合同

共同控制合同不等于共同算法。当前缺少的是统一的 `RunView`、`ControlCommand`、终态、checkpoint、Effect 和 Artifact 语义，而不是缺少一个把所有代码搬进去的新巨型 Engine。

### 根因 C：Workflow 的表达模型与 Autonomous 的耐久模型没有连接

一个擅长“怎么组织 Agent”，另一个擅长“如何保证状态、Effect 和恢复可信”。二者尚未通过稳定的端口连接，导致编排丰富性和生产可靠性无法同时成立。

### 根因 D：主动触发尚未走完同一条准入和权限路径

如果定时器、事件监听器或 Agent 自己能绕过人工入口直接创建工作，就会形成第二套安全模型。主动工作必须与人工任务共享 admission、policy、budget、approval、journal 和 kill switch。

### 根因 E：产品能力和内部技术分类仍然靠命令堆叠暴露

用户真正关心的是“找谁做、做什么、做到哪、能否停、为何需要我批准”，而不是内部使用了 ACP、CLI、Workflow 还是 Team Coordinator。技术入口可以保留给高级用户，但默认产品表面需要围绕项目、员工、团队、任务、审批和审计收敛。

### 根因 F：自动化正确性证据很强，Owner 真实体验证据仍不足

文件 Journal、进程 Actor、Feishu SDK、provider CLI 和本机资源共同决定真实体验。单元测试不能替代当前账号下的飞书端到端、限流、断网、移动端卡片、重启和恢复验证；但这些验证用于发现并修复问题，不用于制造额外发布层级。

---

## 4. 目标架构：一个控制面，三条执行通道

```text
Feishu/Lark Events + Human Commands + Approved Triggers
                         |
                  Effective Context
                         |
                 Immutable RouteDecision
                         |
              Task Read/Control Plane
                         |
       +-----------------+------------------+
       |                 |                  |
 Direct Lane      Protected Strategy   Development Lane
 explicit Agent      Deep / Spec       WT / WF / Team / Slock
 no planner hop      existing engine      typed orchestration
       |                 |                  |
       +------ RunView / Control Ports -----+
                         |
          Agent Backend Capability Catalog
                         |
       Session Factory / Backend Drivers
          ACP | CLI | approved remote
                         |
      Policy + Budget + Effect Dispatch Gate
                         |
       Journal + Blob + Vault + Projections
                         |
        Cards / Audit / Recovery / Metrics
```

### 4.1 Direct Lane

Direct Lane 负责普通 `/codex`、`/coco`、`/claude` 等显式 Agent 编程会话：

- 显式选择永远优先，不经过 LLM 路由。
- 只进行本地 capability 与缓存 discovery snapshot 校验；不得同步探活或更新。
- 保持原会话启动、续接、取消、重试和主卡更新。
- 统一控制面通过事件或轻量适配器观察它，而不是把它编译成工作流。
- Agent catalog 不可用时，兼容入口应 fail clearly；不得静默换成另一个 provider。

### 4.2 Protected Strategy Lane

Deep 和 Spec 继续拥有自己的 manager、阶段算法、提示、持久化和 UI：

- 第一阶段只实现 `RunViewAdapter` 和 `RunControlAdapter`。
- 任何内部生命周期迁移都需要单独工程检查点、字符化测试和回滚开关。
- 统一编排器可以把一个完整 Deep/Spec run 当作黑盒节点，但不能越过其 manager 操纵内部步骤。
- Deep/Spec 的直接用户入口不依赖新 Orchestration Runtime 可用。

### 4.3 Development Lane

Worktree、Workflow、Team、Slock 承担新模型试点，但职责要拆清：

- **Workflow**：面向用户的可视化/自然语言编排入口，以及兼容 JS DSL 前端。
- **Team v2**：耐久的多 Employee 协作执行器与动态协调器。
- **Slock**：群聊任务识别、选择性唤醒、参与者协商、兼容交互入口。
- **Worktree**：代码隔离、验证、合并这一类有明确边界的执行节点。
- **Orchestration Core**：唯一的计划、节点、依赖、Artifact、预算、Effect 和恢复事实模型。

Slock 不再发展第二个调度内核；Workflow 不再自有另一套真相 Journal；Team 不再自有产品任务列表；Worktree 不成为默认任务容器。它们通过端口组合，但仍保留各自价值。

### 4.4 主动工作不是第四个 Engine

主动工作是任务的**来源和授权方式**，不是另一套执行算法：

```text
TriggerDefinition
    -> TriggerOccurrence (dedupe_key, observed_at)
    -> AdmissionDecision
    -> GoalTemplate + AuthorityEnvelope
    -> Direct single-node / Workflow plan / Team run
    -> Policy check for every external Effect
    -> Result / Exception / Approval / Digest
```

这样同一个目标既可由人立即触发，也可由 cron、飞书事件、代码仓库事件或人工批准后的 standing order 触发；后续执行、停止、审计和恢复完全一致。

---

## 5. 关键方案比较与推荐

### 5.1 是否把所有 Engine 合并

| 方案 | 好处 | 弊端 | 结论 |
| --- | --- | --- | --- |
| 保持所有 Engine 完全独立 | 短期风险最低 | 状态、取消、恢复、UI 和审计持续分叉 | 不足以支撑平台化 |
| 一次性重写为统一 Engine | 概念整齐 | 成熟路径回归风险极高，迁移周期长，容易造巨型抽象 | 明确拒绝 |
| 共享控制合同，内部算法独立 | 可先统一可见性、控制和安全；可渐进迁移 | 需要维护 adapter 和双轨期 | **推荐** |

### 5.2 编排表示：继续 JS、Python DSL，还是类型化 IR

| 方案 | 表达力 | 可验证/可恢复 | 迁移成本 | 结论 |
| --- | --- | --- | --- | --- |
| JS 脚本继续作为唯一事实 | 高 | 较弱；动态路径执行前不完整 | 最低 | 保留为兼容前端，不再作为耐久事实 |
| 新建 Python DSL | 中高 | 取决于实现 | 高，且与 JS 重复 | 不推荐 |
| 冻结 JSON/Dataclass IR，前端编译到 IR | 有限但可扩展 | 强；便于验证、投影、恢复 | 中高 | **推荐为 canonical plan** |

IR 不应尝试静态表达无限动态行为。动态任务通过受限节点实现：

- `decision`：根据结构化分类选择预声明分支。
- `map`：对有上限的输入集合展开节点。
- `loop`：有最大轮数、预算和完成判据。
- `coordinator`：允许 LLM 提议增量 patch，但 patch 必须重新验证并写入 Journal。

### 5.3 中央协调还是 Agent 点对点自治

| 方案 | 优点 | 风险 | 适用 |
| --- | --- | --- | --- |
| 完全点对点 | 灵活、涌现性强 | 难以终止、审计、去重和预算；失败归因困难 | 研究/沙盒 |
| 单一中央 LLM 协调器 | 易观察、易控制 | 单点判断偏差、上下文瓶颈 | 小团队短任务 |
| 耐久协调器 + 有界 handoff | 有统一事实源，同时允许局部自治 | 实现复杂度较高 | **GhostAP 推荐** |

Team Coordinator 负责计划 patch、资源和收敛；Employee 之间允许有界 handoff。所有 handoff、任务声明和结果都进入同一个 Run projection。

### 5.4 Backend 插件是配置清单还是任意 Python 插件

首阶段推荐“声明式 descriptor + 仓库内受信 driver registry”：

- descriptor 描述能力、传输、模型发现和健康探测。
- driver 实现受控 `SyncSession` 合同。
- CI 运行一致性套件。
- 不允许从聊天消息下载并在主进程执行第三方 Python 插件。

以后若需要外部生态，优先使用隔离进程或远程协议适配器。任意 in-process 插件会直接突破 GhostAP 当前的凭据、Shell 和主 Bot 信任边界。

### 5.5 主动触发：cron、事件还是 Agent 自轮询

| 方式 | 好处 | 问题 | 建议 |
| --- | --- | --- | --- |
| Agent 自己循环轮询 | 实现看似简单 | 无明确生命周期、成本不可控、重启语义差 | 禁止作为生产默认 |
| 仅 cron | 易理解 | 无法响应业务事件，补跑语义复杂 | 作为首个 adapter |
| 统一 TriggerOccurrence | cron/事件/webhook 共用准入、去重和审计 | 初始建模成本高 | **推荐** |

### 5.6 文件内核还是立即多副本

当前产品约束和已有 ADR 适合继续以单机、文件持久化 profile 为默认：

- 优点：部署简单、现有 Journal/Vault/Blob 投入可复用、故障域清晰。
- 缺点：单写者吞吐、单机可用性、跨副本租约和 rollback resistance 存在天花板。

当前明确采用单 Owner、单主机部署，不应为多副本提前重写。若未来真实容量或可用性需求超过单机边界，多副本应另立计划，要求外部协调、共享耐久存储、KMS/workload identity 和故障注入；不能通过在共享目录上启动两个进程来声称实现。

---

## 6. 能力边界与天花板

### 6.1 当前已知边界

| 维度 | 当前边界 | 含义 |
| --- | --- | --- |
| 部署 | 内置单主机 profile | 不能声称跨机房或多副本线性一致 |
| Employee 可见规模 | 默认配置为 8 | 是安全默认值，不是压测得出的容量上限 |
| Team run | turn 12、assignment 32、fanout 4、handoff 8 | 是防失控硬边界，不代表质量最优点 |
| Workflow | host 侧总 Agent 调用上限 200；loop/generate 上限 50 | 是单 run 防爆炸上限，不是推荐规模 |
| 状态 | Journal/Blob/Vault 为本地文件事实源 | 吞吐和可用性受单写者、磁盘和单机故障域限制 |
| 外部 Effect | 可锚定、可审计，但远端副作用通常不可事务回滚 | 恢复时必须处理 unknown/committed，不能假装 exactly-once |
| Agent 质量 | 依赖 provider/model/上下文/提示和外部服务 | 平台只能验证合同、比较结果和限制风险，不能保证 LLM 正确 |
| Owner 体验证明 | 缺完整飞书端到端清单 | 自动化测试通过后可直接体验，但清单中的失败必须继续修复 |

### 6.2 本计划完成后的近程能力

完成对应工程检查点后，当前 Owner 可直接获得：

- 一个用户仍可直接选择任意已注册 Agent 进行低摩擦编程。
- 新 Backend 可以通过一个 family/binding 声明、driver 和 conformance suite 接入。
- Workflow/Team/Slock/Worktree 的任务在重启后有可信状态，并能从统一入口查看、停止和恢复。
- cron、飞书事件和批准的外部事件可以创建去重的主动任务。
- 每个主动任务都有 owner、authority、预算、截止时间、通知策略和 kill switch。
- 本机端到端清单能诚实区分“通过、失败、未测试”，并可一键关闭新子系统而不影响 Direct/Deep/Spec。

### 6.3 单机架构的实际天花板

不应在没有测量时承诺固定并发数字。单机 profile 的上限由以下最小项决定：

```text
min(
  provider rate/concurrency limit,
  local ACP/CLI process capacity,
  Journal single-writer throughput,
  Feishu API/WebSocket rate limit,
  CPU/RAM/file descriptor capacity,
  card update coalescing capacity
)
```

本计划要求在需要提高默认上限时通过负载实验生成容量曲线，而不是写死“支持 N 个并发 Agent”。默认 8 员工先作为配置上限直接使用；只有 Owner 实际需要更高上限时，才以 queue wait、Journal fsync、进程数、卡片限流和 provider 429 证据调整。

### 6.4 不能被软件抽象消除的边界

- 远端 Agent 或 Shell 已执行的副作用通常不能自动回滚。
- 不同 provider 不保证相同工具、上下文、模型名称或可续接语义。
- Agent 之间自然语言交接会丢失信息；必须使用 Artifact schema 和验收标准降低风险。
- LLM 协调器不能替代权限系统；“它认为安全”不是授权。
- 无限循环、无限 fanout 和无预算主动任务不能成为生产能力。
- 在 file-only 单机 profile 下，无法诚实承诺跨副本共识、区域容灾或外部防回滚。

### 6.5 Hardened Profile 的长期天花板（不进入当前实施路线）

如果未来目标是跨团队、高可用、数百长期 Employee 或跨主机编排，需要单独立项：

- 外部一致性存储或日志。
- 分布式 lease/fencing。
- 外部 KMS、workload identity 和不可伪造 attestation。
- Agent worker 容器/沙箱和网络 egress policy。
- 多租户配额、账单、数据驻留和审计导出。
- 多副本故障注入与灾难恢复演练。

这些能力不是本计划单机阶段的“顺手增强”，而是另一种部署产品。

---

## 7. 路线图、依赖和工程检查点

### 7.1 总依赖图

```text
Foundation Track（既有产品收敛计划）
  ├─ 产品/安全/入口真相
  ├─ EffectiveContext + RouteDecision
  ├─ Task read/control plane
  └─ readiness/backup/本机验证
             |
             v
Phase 0：成熟路径保护 + 当前正确性止血
             |
             +----------------------+
             |                      |
             v                      v
Phase A：Agent Connectivity   Phase B1：统一 Run/IR 编译合同
             |                      |
             +-----------+----------+
                         v
                Phase B2：耐久编排执行
                         |
                         v
                 Phase C：主动任务闭环
                         |
                         v
                Phase D：评估与自适应
                         |
                         v
                Phase E：本机体验、恢复与运维
```

Phase A 的静态 catalog 和 Phase B 的只读 RunView 可以并行；任何主动触发都必须等待耐久 Run、Policy、Effect 和 kill switch 可用。

### 7.2 工程检查点

这些检查点是实现依赖和正确性边界，不是发布层级。每个检查点通过后，对应功能立即在当前 Owner 的正常入口可见；失败时修复或使用紧急回退开关，不创建 soak、allowlist 或晋级流程。

| Checkpoint | 必须回答的问题 | 完成条件 | 未完成时 |
| --- | --- | --- | --- |
| CP-P0：保护基线 | Direct/Deep/Spec 的真实合同是什么 | 字符化测试通过；无额外 LLM hop；已知“显示成功但实际未执行”问题被修复或禁用 | 不迁移成熟路径 |
| CP-A：Backend Catalog | catalog 是否比散落分支更真实 | 所有内建 Backend 通过 conformance；显式选择不会静默 fallback | 保留兼容工厂为 SSOT |
| CP-B-Compile：IR 编译 | IR v2 能否在零 dispatch 下被确定性验证 | schema、compiler、预算、单一生成源、unsupported fallback 和零外部调用测试通过 | v1 继续可用，v2 不执行 |
| CP-B-Execute：耐久执行 | 恢复后是否会重复或遗漏 Effect | kill/restart/unknown Effect/cancel acknowledgment 和 schema migration 测试通过 | IR 不驱动外部写操作 |
| CP-C-Assist：主动只读 | Trigger 是否能安全地产生只读任务 | occurrence 去重、TTL、预算、owner、data egress、pause/kill、quiet hours 全部通过 | 仅允许手工触发 |
| CP-C-Write：Owner 确认写入 | 是否可以执行 Owner 明确批准的可逆写操作 | 每 Run 确认、risk gate、nonce、审计、Effect 锚定和例外通知通过 | 保持 assist/read-only |
| CP-E：Owner 运维闭环 | 已开放能力是否能持续使用和恢复 | 每项开放前最小 smoke 通过；开放后清单持续记录桌面/移动端、重启/备份恢复和紧急回退 | 记录为需修复缺陷或回退对应子系统，不伪报完成 |

CP-E 是跨功能的持续运维清单，不是第二道总发布门；它不能延迟已经通过自身检查点和最小 smoke 的能力。

### 7.3 Foundation Track 的使用方式

不要复制既有收敛计划的 26 个任务。实施本计划前，先从该计划建立一张依赖账本：

| 本计划依赖 | 对应既有任务 | 最低要求 |
| --- | --- | --- |
| 产品与信任边界 | Task 1–8 | 明确 trusted engineering host、入口权限、Employee enablement 和 topic routing |
| 统一产品控制面 | Task 9–13 | action catalog、effective context、RouteDecision、task read/control |
| 共享运行合同 | Task 14–20 | backend catalog/session request、run lifecycle、WT/WF 恢复正确性 |
| 运维与本机体验 | Task 21–24、Task 26 的本机部分 | legacy quarantine、doctor、backup、可选 sandbox、本机切换与回退 |

原 Task 25 的签名 Beta Gate 不实施；Task 26 中按租户或阶段切流的内容删除，只保留当前部署的一次启用、兼容回退和验证。

若既有任务尚未实施，执行者应直接使用既有计划中的测试和文件步骤。本计划只在以下地方覆盖它：

- Direct/Deep/Spec 为保护路径；不能为了“统一”修改内部算法。
- Workflow/Worktree/Team/Slock 是新合同首批试点。
- 任何发现的新高正确性或安全问题必须在扩展能力前修复。

---

## 8. Phase 0 — 保护成熟路径并修正当前事实

**目标：** 先消除“UI/文案承诺与实际执行不一致”，建立后续重构不可突破的基线。

**预计投入：** 15–25 engineer-days。这里的工作是后续平台化的准入门，不是可选 polish。

### Task 0.1：建立执行通道完成度与产品合同

**Files:**

- Create: `docs/execution-lanes.md`
- Create/complete: `src/feishu/product_catalog.py`
- Create/complete: `tests/test_product_action_catalog.py`

既有收敛计划 Task 9 是本任务硬前置：若 catalog/test 尚不存在，先按该任务创建；若已存在，只扩展完成度字段。不得在缺文件时跳过实现却无条件运行验证。

**Implementation:**

1. 定义固定枚举 `mature`、`developing`、`not_implemented`；运行时健康另用 `available/degraded/unavailable`，不要混成发布状态机。
2. 将 direct、deep、spec 标为 `mature`；worktree、workflow、team、slock 在合同未完成时标为 `developing`。
3. 完成度只影响说明文字和支持预期，不控制 Owner 的菜单曝光、租户准入或分批发布。
4. 已有实现和新完成能力都显示在 Owner 菜单中；未满足执行前置的动作显示具体阻塞原因，不使用隐藏功能代替真实校验。
5. 删除/修正文案中已经退役但仍被宣传的 `/goal` 等入口；在新的主动入口完成前不得暗示可用。
6. SMART/Slock 自动激活仍不得截获 Direct/Deep/Spec 明确命令；这是路由安全，不是发布策略。

**Tests first:**

```python
def test_direct_deep_spec_are_mature_by_default(): ...
def test_implemented_developing_lane_is_visible_to_owner(): ...
def test_completion_label_does_not_gate_owner_access(): ...
def test_developing_lane_never_auto_activates_over_explicit_command(): ...
def test_retired_goal_command_is_not_advertised(): ...
```

**Acceptance:**

- SMART 或 Slock 自动激活不能截获 direct/deep/spec 明确命令。
- Owner 能看见已实现能力及其真实完成度；完成后不需要额外 allowlist 或 opt-in。
- 现有命令别名和卡片动作保持兼容。

**Verify:**

```bash
uv run pytest tests/test_product_action_catalog.py -q
```

**Commit:** `feat(product): expose execution lane completion`

### Task 0.2：为 Direct Lane 建立纵向合同测试

**Files:**

- Create: `tests/contracts/test_direct_programming_lane.py`
- Create: `tests/contracts/test_protected_execution_lanes.py`
- Create: `tests/helpers/session_call_recorder.py`
- Create: `scripts/benchmark_direct_lane.py`
- Modify: `docs/testing.md`

**Implementation:**

1. 从飞书命令/当前 programming mode 一直测试到 session factory 的真实请求。
2. 对 Coco、Claude、Aiden、Codex、Gemini、Traex 和 TTADK representative backend 参数化。
3. 记录每次请求中的 backend、model、cwd、chat/project/thread session key、tool filter 和 prompt。
4. 断言显式 Agent 请求只产生一个目标 session prompt，不调用 classifier、planner、reviewer 或 coordinator。
5. 断言续聊复用同一个 session key，取消仍到达实际 session。
6. benchmark 只记录基线分布；准入条件是新增 catalog 不增加网络/LLM hop，不在尚无稳定环境时武断设置毫秒阈值。
7. 对 Deep/Spec 参数化记录 selected provider/model、provider-local retry、session factory 请求和既有取消/恢复声明，作为 Phase A 共享工厂迁移门。

**Tests first:**

```python
def test_explicit_codex_uses_exactly_one_backend_prompt_and_no_planner(): ...
def test_direct_lane_preserves_chat_project_thread_session_key(): ...
def test_direct_cancel_reaches_selected_session(): ...
def test_backend_start_failure_does_not_persist_selected_project_state(): ...
def test_deep_and_spec_preserve_provider_model_and_retry_contract(): ...
```

**Acceptance:**

- 新架构上线前后 recorder 中远程调用拓扑相同。
- 显式选择失败时给出明确错误，不自动改用另一 Backend。

**Verify:**

```bash
uv run pytest tests/contracts/test_direct_programming_lane.py tests/contracts/test_protected_execution_lanes.py -q
uv run python scripts/benchmark_direct_lane.py --runs 20
```

**Commit:** `test(programming): freeze direct agent lane contract`

### Task 0.3：修正 Claude CLI 模型选择的真实性

**Why now:** 当前普通 Claude CLI 会话的 model 选择可能在 UI 中显示成功，但 `SyncClaudeCLISession` 启动参数没有真实携带所选 model；这是平台 catalog 上线前必须修复的纵向合同缺口。

**Files:**

- Modify: `src/agent_session/claude_cli.py`
- Modify: `src/agent_session/factory.py`
- Modify: `src/acp/startup_utils.py`
- Modify: `src/acp/manager.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/project/manager.py`
- Test: `tests/test_claude_cli_args.py`
- Test: `tests/test_switch_model.py`
- Test: `tests/test_claude_1m_context.py`

**Implementation:**

1. 先用真实 argv 测试证明当前 model 丢失。
2. 明确二选一产品合同：
   - 如果当前 Claude CLI 版本支持显式 model，则全链路传递并验证 argv。
   - 如果不支持，则从 capability 和 UI 中移除 model selection，禁止显示“切换成功”。
3. 1M 上下文能力必须绑定真实 CLI 参数/环境证据，不能只测试 ACP provider。
4. 修正 `SystemHandler._enter_mode_with_acp_model` 一类先写 `project.acp_tool_name/acp_model_name` 再启动的路径：session 激活成功后再通过 ProjectManager 原子提交 selected backend/model；启动失败保留原状态。

**Tests first:**

```python
def test_selected_claude_model_reaches_real_cli_argv(): ...
def test_claude_1m_selection_reaches_real_cli_environment(): ...
def test_failed_claude_restart_keeps_previous_project_selection(): ...
```

**Acceptance:**

- UI 显示的 model 与真实进程 argv/env 一致。
- 不支持的能力不可被选择。

**Verify:**

```bash
uv run pytest tests/test_claude_cli_args.py tests/test_switch_model.py tests/test_claude_1m_context.py -q
```

**Commit:** `fix(agent-session): make claude model selection truthful`

### Task 0.4：修正 Workflow 的执行绑定与 Reviewer 承诺

**Why now:** 当前选中的 `tool_model_map` 可能没有进入 Engine 实际使用的 `WorkflowProject`；Reviewer 选择主要进入脚本提示，并不保证独立 Reviewer 被调用。

**Files:**

- Create: `src/workflow_engine/run_spec.py`
- Modify: `src/workflow_engine/models.py`
- Modify: `src/workflow_engine/engine.py`
- Modify: `src/feishu/handlers/workflow.py`
- Modify: `src/workflow_engine/script_gen.py`
- Test: `tests/test_workflow_execution_bindings.py`
- Test: `tests/test_workflow_reviewer_contract.py`

**Implementation:**

1. 引入冻结、不可变的 `WorkflowRunSpec`，由 Handler 一次性传入 Engine，至少包含 orchestrator、reviewers、每个 tool 的 model、task、chat/topic、budget 和 deadline。
2. Engine 禁止重新创建缺少绑定的 project。
3. Reviewer 产品语义做明确选择：
   - 选择了独立 Reviewer，就必须创建独立调用并持久记录证据；或
   - 将 UI 改名为“允许运行的工具”，不再宣称 Reviewer。
4. Auto reviewer 必须在 RunSpec 中显式表达，不能依靠空列表推断。
5. 先与当前工作区正在进行的 Workflow 卡片修复对齐，避免覆盖并行修改。

**Tests first:**

```python
def test_selected_model_map_reaches_every_agent_call(): ...
def test_each_explicit_reviewer_is_independently_invoked(): ...
def test_auto_reviewer_is_explicit_in_run_spec(): ...
```

**Acceptance:**

- 卡片选择、确认摘要、执行 trace 和最终报告中的 backend/model/reviewer 完全一致。
- 未实际发生的 Reviewer 不得出现在“已评审”记录中。

**Verify:**

```bash
uv run pytest tests/test_workflow_execution_bindings.py tests/test_workflow_reviewer_contract.py -q
```

**Commit:** `fix(workflow): bind selected agents to actual execution`

### Task 0.5：修正 Worktree 终态、超时和真实评审

**Files:**

- Modify: `src/worktree_engine/dispatcher.py`
- Modify: `src/worktree_engine/manager.py`
- Modify: `src/worktree_engine/review_adapter.py`
- Test: `tests/test_worktree_dispatcher_timeout.py`
- Create: `tests/test_worktree_terminal_truth.py`
- Create: `tests/test_worktree_review_contract.py`

**Implementation:**

1. timeout 必须调用每个运行 session 的 `cancel()`，并在有界时间内返回 cancellation acknowledgment。
2. 不再依靠 `ThreadPoolExecutor` context 的默认 `shutdown(wait=True)` 伪装硬超时。
3. 任一必要 unit failed/cancelled 时，journey 不得进入 `COMPLETED`，也不得自动 merge。
4. Reviewer 必须消费真实 diff/test/findings 并输出结构化 verdict；空 findings 不等价于通过。
5. 卡片披露 WT 分支优先解决冲突的现有产品规则。

**Tests first:**

```python
def test_pool_timeout_returns_within_bound_and_cancels_session(): ...
def test_failed_unit_prevents_journey_completed_and_merge(): ...
def test_review_consumes_real_findings_before_completion(): ...
```

**Acceptance:**

- wall-clock timeout 是硬边界。
- 高层终态与所有必要 unit、review、merge 事实一致。

**Verify:**

```bash
uv run pytest tests/test_worktree_dispatcher_timeout.py tests/test_worktree_terminal_truth.py tests/test_worktree_review_contract.py -q
```

**Commit:** `fix(worktree): make timeout and terminal state truthful`

### Task 0.6：局部修正 Spec completion fail-open，不重写 Spec

**Why now:** 保护成熟路径不代表保留已确认的高正确性缺陷。模型明确返回 `FAIL/GOAL_NOT_MET` 时，不能因为缺少高置信 blocker 或返回非 JSON 就自动判通过。

**Files:**

- Modify: `src/spec_engine/adaptive_review.py`
- Modify: `src/spec_engine/engine.py`
- Test: `tests/test_adaptive_review_pipeline.py`
- Create: `tests/test_adaptive_review_completion_control.py`

**Implementation:**

1. 将显式 `FAIL`、`GOAL_NOT_MET`、解析失败和缺少 verdict 设为 fail-closed。
2. 区分“评审失败”和“传输/格式失败”；后者进入有界重试或人工确认。
3. 保留 Spec 现有阶段、UI、provider、model 和 retry 行为。
4. 不把 Spec 内部步骤迁移到新 IR。

**Tests first:**

```python
def test_goal_not_met_without_suggestions_fails_closed(): ...
def test_non_json_review_cannot_mark_completion_passed(): ...
def test_existing_spec_stage_and_retry_contract_is_unchanged(): ...
```

**Acceptance:**

- 明确失败永远不能因为空建议被转成通过。
- 现有 Spec 主流程回归测试全部通过。

**Verify:**

```bash
uv run pytest tests/test_adaptive_review_pipeline.py tests/test_adaptive_review_completion_control.py tests/test_spec_engine.py -q
```

**Commit:** `fix(spec): fail closed on invalid completion verdicts`

### Task 0.7：封闭协调/分类辅助 Agent 的工具权限

**Files:**

- Modify: `src/autonomous/team/coordinator.py`
- Modify: `src/slock_engine/intent_router.py`
- Modify: `src/slock_engine/discussion_manager.py`
- Create: `tests/autonomous/security/test_coordinator_tool_filter.py`
- Create: `tests/test_slock_auxiliary_session_permissions.py`

**Implementation:**

1. Team Coordinator 的 JSON decision session 使用 deny-all tool filter；只允许文本/结构化返回。
2. Slock intent classifier、NLI 和讨论摘要同样禁止项目工具和 Shell。
3. `auto_approve=True` 不能单独作为安全策略；session purpose 必须决定权限 profile。
4. 如果某个协调任务确需读仓库，使用单独的 read-only context collector，结果作为 Artifact 注入，不能给 coordinator 任意工具。

**Tests first:**

```python
def test_team_coordinator_cannot_call_project_tools(): ...
def test_slock_classifier_cannot_call_shell_or_write_tools(): ...
def test_read_only_context_is_passed_as_data_not_tool_authority(): ...
```

**Acceptance:**

- prompt injection 无法让协调/分类 Agent 获得项目写权限。
- Employee 实际执行权限仍由原有 capability + ACL 组合决定。

**Verify:**

```bash
uv run pytest tests/autonomous/security/test_coordinator_tool_filter.py tests/test_slock_auxiliary_session_permissions.py -q
```

**Commit:** `fix(security): deny tools to coordination-only sessions`

### Checkpoint CP-P0 验证

```bash
uv run pytest tests/contracts/test_direct_programming_lane.py -q
uv run pytest tests/test_deep_engine.py tests/test_deep_completion_guard.py -q
uv run pytest tests/test_spec_engine.py tests/test_spec_task_persistence.py tests/test_adaptive_review_pipeline.py -q
uv run pytest tests/test_workflow_*.py tests/test_worktree_*.py -q
uv run pytest tests/ -q -m "not slow"
uv run ruff check src/agent_session src/deep_engine src/spec_engine src/workflow_engine src/worktree_engine src/autonomous/team src/slock_engine
uv run python scripts/test_inventory.py tests/
```

检查报告必须明确区分：

- Deep 的进程内 resume 与 crash-safe recovery。
- Workflow 的调用 cache 与真正的 lifecycle Journal。
- Employee Channel 的 process fallback 与 OS filesystem isolation。
- 本地 mock 通过与当前 Owner 飞书账号端到端通过。

---

## 9. Phase A — Agent Connectivity Platform

**目标：** 把“多处硬编码支持多个工具”收敛为“一个可验证的后端能力目录 + 一个兼容会话工厂”，同时保持 Direct Lane 最短路径。

**预计投入：** 25–40 engineer-days。

### Task A1：定义 Backend 能力和会话请求的中立类型

**Files:**

- Create: `src/agent_session/capabilities.py`
- Create: `src/agent_session/identity.py`
- Create: `src/agent_session/environment.py`
- Create: `src/agent_session/options.py`
- Create: `src/agent_session/request.py`
- Modify: `src/agent_session/protocol.py`
- Test: `tests/test_backend_capabilities.py`
- Test: `tests/test_session_request.py`

**Required types:**

```python
class TransportKind(StrEnum):
    ACP = "acp"
    CLI = "cli"
    REMOTE = "remote"

class Capability(StrEnum):
    STARTUP_MODEL_SELECTION = "startup_model_selection"
    LIVE_MODEL_SWITCH = "live_model_switch"
    PERSISTENT_CONTEXT = "persistent_context"
    SESSION_ID_LOAD = "session_id_load"
    STRUCTURED_EVENTS = "structured_events"
    USAGE = "usage"
    IMAGE_INPUT = "image_input"
    IMAGE_OUTPUT = "image_output"
    TOOL_PERMISSION_CALLBACK = "tool_permission_callback"
    TOOL_FILTER_ENFORCED = "tool_filter_enforced"
    NETWORK_EGRESS_ENFORCED = "network_egress_enforced"
    CANCEL_REQUEST = "cancel_request"
    CANCEL_ACK = "cancel_ack"
    HEALTH_PROBE = "health_probe"

class ToolMediation(StrEnum):
    NONE = "none"
    DECLARED_ONLY = "declared_only"
    LEXICAL_HOST = "lexical_host"
    ACP_CALLBACK = "acp_callback"
    OS_SANDBOX = "os_sandbox"

class PermissionProfile(StrEnum):
    DENY_ALL = "deny_all"
    READ_ONLY = "read_only"
    PROJECT_SCOPED_WRITE = "project_scoped_write"
    EMPLOYEE_POLICY = "employee_policy"

class SessionPurpose(StrEnum):
    DIRECT = "direct"
    ENGINE = "engine"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    EMPLOYEE = "employee"

@dataclass(frozen=True)
class BackendCapabilities:
    supported: frozenset[Capability]
    transport: TransportKind
    stateless: bool
    tool_mediation: ToolMediation

@dataclass(frozen=True)
class BackendBinding:
    binding_id: str
    family_id: str
    parameters: tuple[tuple[str, str], ...]
    effective_capabilities: BackendCapabilities
    trusted_argv_ref: str | None

@dataclass(frozen=True)
class SessionRequest:
    binding: "BackendBinding"
    model_id: str | None
    purpose: SessionPurpose
    cwd: Path
    session_key: str
    permission_profile: PermissionProfile
    environment_ref: "EnvironmentRef | None"
```

`identity.py` 定义 Backend family/binding ids 和上述 `BackendBinding`；`options.py` 定义中立 `ToolDescriptor/ModelDescriptor`；`environment.py` 定义不可序列化、`repr=False` 的 `EnvironmentRef`。真正环境由 production composition/Vault 注入的 provider 在 driver 边界解析。`SessionRequest` 不能携带明文员工凭据、任意 env mapping 或可以被日志序列化的 secret。`PermissionProfile` 替代含义模糊的 `auto_approve: bool`；driver 必须证明它能强制执行所选 profile。

**Tests first:**

```python
def test_coordinator_request_requires_deny_all_tool_filter(): ...
def test_capabilities_are_serializable_for_cards_and_diagnostics(): ...
def test_declared_only_tool_filter_is_not_exposed_as_enforced(): ...
def test_environment_ref_never_serializes_or_reprs_secret_material(): ...
```

**Acceptance:**

- 类型不依赖 Feishu Handler、Workflow 或 Autonomous UI。
- 不能使用 `getattr` 猜测关键安全能力。

**Verify:**

```bash
uv run pytest tests/test_backend_capabilities.py tests/test_session_request.py -q
```

**Commit:** `feat(agent-session): define backend capability contracts`

### Task A2：建立唯一 Backend Catalog

**Files:**

- Create: `src/agent_session/catalog.py`
- Create: `src/agent_session/builtin_backends.py`
- Modify: `src/acp/providers/__init__.py`
- Modify: `src/workflow_engine/tool_registry.py`
- Modify: `src/feishu/action_registry.py`
- Modify: `src/feishu/session_hub.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/feishu/handlers/programming.py`
- Modify: `src/feishu/handlers/system.py`
- Modify: `src/mode/manager.py`
- Modify: `src/config/settings.py`
- Modify: `src/agent/intent_recognizer.py`
- Modify: `src/worktree_engine/tool_discovery.py`
- Modify: `docs/acp_provider_guide.md`
- Test: `tests/test_backend_catalog.py`
- Test: `tests/test_backend_catalog_consumers.py`

**Required family/binding descriptors:**

```python
@dataclass(frozen=True)
class BackendFamily:
    family_id: str
    display_name: str
    aliases: tuple[str, ...]
    driver_id: str
    capability_ceiling: BackendCapabilities
    binding_schema_id: str
    model_discovery_id: str | None
    availability_probe_id: str
    default_model: str | None
    required_config_keys: tuple[str, ...]
```

**Implementation:**

1. 先把当前内建后端逐一录入 catalog，不改变命令别名。
2. 区分静态 `BackendFamily` 与运行时 `BackendBinding`：一个 ACP driver 可服务多个 provider；TTADK tool、TUI2ACP adapter/custom argv 作为经过 schema/受信参数规则验证的 binding 参数，不伪装成无限静态 descriptor。
   `parameters` 只允许非机密枚举/标识；custom argv 和凭据通过受信引用解析，不把原文放入 catalog、卡片或日志。
   Family 的 capability 只是上限；每个 binding 经 discovery/conformance 得到 `effective_capabilities`，路由和 UI 只能消费后者。
3. provider、生产 `ws_client` manager 组装、programming/system Handler、mode manager、配置 validator、manager hub、合法工具校验、Workflow 工具列表和 Worktree discovery 从 catalog 派生。
4. `SessionManagerHub` 当前不是生产组装 SSOT；在生产 `ws_client` 完成迁移前不得用 hub 通过推断整体完成。
5. 原 hardcoded 集合在迁移期保留只读一致性断言，随后删除。
6. catalog 查询必须为本地 O(1)，不得做网络、npm update 或 session 启动。
7. 员工身份不进入 catalog；Employee 只引用 `BackendBinding + model_id`。

**Tests first:**

```python
def test_every_builtin_backend_binding_has_one_family_and_unique_id(): ...
def test_all_public_command_aliases_resolve_through_catalog(): ...
def test_catalog_lookup_has_no_probe_install_or_model_discovery_side_effect(): ...
def test_workflow_worktree_and_direct_lists_share_same_backend_ids(): ...
def test_production_ws_client_and_programming_modes_are_derived_from_catalog(): ...
def test_parameterized_ttadk_and_tui_bindings_validate_against_family_schema(): ...
def test_binding_effective_capabilities_cannot_exceed_family_ceiling(): ...
```

**Acceptance:**

- 新增内建 Backend 的静态身份只改一个 family/binding 声明。
- 所有消费者的 family/binding id 集合无漂移。

**Verify:**

```bash
uv run pytest tests/test_backend_catalog.py tests/test_backend_catalog_consumers.py -q
```

**Commit:** `feat(agent-session): add canonical backend catalog`

### Task A3：用 Driver 收敛会话工厂，保留兼容入口

**Files:**

- Create: `src/agent_session/driver.py`
- Create: `src/agent_session/drivers/acp.py`
- Create: `src/agent_session/drivers/claude_cli.py`
- Create: `src/agent_session/drivers/ttadk_cli.py`
- Modify: `src/agent_session/factory.py`
- Modify: `src/acp/session_factory.py`
- Modify: `src/acp/startup_utils.py`
- Modify: `src/acp/manager.py`
- Test: `tests/test_backend_driver_contract.py`
- Test: `tests/test_session_factory_compatibility.py`

**Required protocol:**

```python
class BackendDriver(Protocol):
    driver_id: str

    def create(self, request: SessionRequest) -> SyncSession: ...
    def validate(self, request: SessionRequest) -> tuple[str, ...]: ...
```

**Implementation:**

1. 将 TTADK/Claude/ACP 特化封装进各自 driver。
2. `src/agent_session/factory.py` 成为唯一新实现。
3. `src/acp/session_factory.py` 在迁移期只做参数转换和委托；不得继续复制分支。
4. `ACPSessionManager` 的 `SessionStartupCoordinator` 同样委托新 driver；manager、普通 factory、engine、reviewer 和 employee-env 五条启动链均纳入 contract。
5. 保留现有 `create_sync_session`、`create_engine_session` 签名直到所有调用方迁移。
6. 利用 Task 0.2 recorder 对比迁移前后 Direct Lane 调用拓扑。

**Tests first:**

```python
def test_legacy_factory_and_new_driver_create_equivalent_direct_sessions(): ...
def test_ttadk_branch_exists_only_inside_ttadk_driver(): ...
def test_explicit_backend_never_falls_back_to_different_driver(): ...
def test_manager_engine_reviewer_and_employee_paths_share_driver_binding(): ...
def test_driver_rejects_model_for_binding_without_startup_selection(): ...
def test_driver_lookup_uses_family_driver_id_and_validates_binding_schema(): ...
```

**Acceptance:**

- 引擎和 Handler 不再出现新的 `startswith("ttadk_")`。
- Direct Lane 没有额外异步调度、LLM 调用或远程探测。

**Verify:**

```bash
uv run pytest tests/test_backend_driver_contract.py tests/test_session_factory_compatibility.py tests/contracts/test_direct_programming_lane.py -q
```

**Commit:** `refactor(agent-session): centralize backend drivers`

### Task A4：将发现、探活和安装/更新彻底分离

**Files:**

- Create: `src/agent_session/discovery.py`
- Modify: `src/agent_session/options.py`
- Modify: `src/acp/helper.py`
- Modify: `src/acp/provider.py`
- Modify: `src/acp/client.py`
- Modify: `src/acp/sync_adapter.py`
- Modify: `src/acp/providers/__init__.py`
- Modify: `src/ttadk/manager.py`
- Modify: `src/ttadk/models.py`
- Modify: `src/feishu/handlers/system.py`
- Test: `tests/test_backend_discovery.py`
- Test: `tests/test_backend_admin_update.py`

**Required result:**

```python
@dataclass(frozen=True)
class DiscoveryResult:
    binding_id: str
    models: tuple[ModelDescriptor, ...]
    source: str
    observed_at: datetime
    fresh_until: datetime
    degraded: bool
    error_code: str | None
```

**Implementation:**

1. list/probe/model discovery 必须是无安装副作用的读操作。
2. npm/CLI 安装、更新和 repair 只能由显式管理员 action 启动并审计。
3. 统一正/负缓存、single-flight 和 stale-while-revalidate 规则。
4. 无模型选择能力的 Backend 直接使用默认行为，不展示空模型卡。
5. model discovery 不得调用 `prompt()`、产生付费推理或获得项目工具权限。若 ACP provider 只能通过 `initialize + new_session` 暴露模型选项，允许创建有界、deny-all、必关闭并带审计的临时 capability session；不得退回不真实的静态列表。
6. 将通用 Tool/Model option 从 `ttadk.models` 移到中立 `agent_session/options.py`，TTADK 只保留 transport 特有字段。
7. 普通选择流与所有 Engine 都消费 catalog 的 startup-model capability/skip-selection 事实。
8. ACP `UsageUpdate` 若 provider 提供则归一化进 session result/trace；不支持时标为 unknown，不能继续静默丢弃或记零。

**Tests first:**

```python
def test_listing_backends_never_installs_or_updates_tools(): ...
def test_model_discovery_is_single_flight_and_reports_freshness(): ...
def test_backend_without_model_selection_skips_model_card(): ...
def test_capability_session_is_deny_all_bounded_and_always_closed(): ...
def test_common_tool_and_model_options_do_not_depend_on_ttadk(): ...
def test_usage_update_is_preserved_or_explicitly_unknown(): ...
```

**Acceptance:**

- 用户打开菜单不会触发最长数十秒的隐式更新。
- UI 能区分 unavailable、degraded、stale 和 unsupported。

**Verify:**

```bash
uv run pytest tests/test_backend_discovery.py tests/test_backend_admin_update.py -q
```

**Commit:** `refactor(agent-session): separate discovery from mutation`

### Task A5：建立 Backend Conformance Kit

**Files:**

- Create: `tests/contracts/backend_conformance.py`
- Create: `tests/contracts/test_builtin_backend_conformance.py`
- Create: `tests/fixtures/backend_drivers/`
- Modify: `scripts/test_inventory.py`
- Create: `docs/backend-driver-contract.md`

**Implementation:**

每个 Backend 必须根据自己声明的 capability 通过条件测试：

- start/send/close 基础合同。
- persistent context 与 session-id load 分开验证；只保存一个 id 不能宣称恢复上下文。
- startup model selection 与 live model switch 分开验证。
- cancel request 与 cancel acknowledgment 分开；声明 acknowledgment 时必须在 deadline 内返回。
- structured events：文本、思考、工具、图片、计划顺序。
- image input 与 image output 分开验证。
- tool permission callback、保存 filter 字段和真正强制拒绝分开验证；CLI 的 no-op setter 不能宣称 enforced。
- 对每个声明支持的 `DENY_ALL`、`READ_ONLY`、`PROJECT_SCOPED_WRITE` profile 做真实拒绝测试；只保存 profile/filter 字段不能通过。
- `NETWORK_EGRESS_ENFORCED` 必须通过目标域安全 allowlist/deny 和旁路测试；opaque CLI 未运行在可证明网络隔离中时不得声明。这里的 allowlist 是数据外发安全边界，不是功能灰度。
- usage：若支持，输入/输出 token 不得丢失。
- health：不得用恒真冒充真实健康。
- stateless CLI：明确降级，不伪装成 ACP 等价能力。

**Tests first:**

```python
@pytest.mark.parametrize("binding", builtin_backend_bindings())
def test_claimed_capabilities_have_executable_evidence(binding): ...

def test_unclaimed_capability_is_never_exposed_to_ui_or_router(): ...
def test_claimed_permission_profiles_are_actually_enforced(): ...
def test_claimed_network_egress_control_blocks_non_allowlisted_domain(): ...
```

**Acceptance:**

- capability 是“可执行证据”，不是营销字段。
- 新 Backend 未通过合同测试不能进入可用 catalog；通过后立即出现在当前 Owner 的选择入口。

**Verify:**

```bash
uv run pytest tests/contracts/test_builtin_backend_conformance.py -q
uv run python scripts/test_inventory.py tests/
```

**Commit:** `test(agent-session): add backend conformance kit`

### Task A6：实现无 LLM 的可解释 Backend Routing Policy

**Files:**

- Create: `src/agent_session/routing.py`
- Modify: `src/agent_session/factory.py`
- Modify: `src/autonomous/team/coordinator.py`
- Modify: `src/workflow_engine/executor.py`
- Test: `tests/test_backend_routing_policy.py`

**Priority order:**

1. 用户显式 backend/model。
2. Employee 持久配置。
3. Workflow/Team RunSpec 的显式绑定。
4. role 所需 capability + Owner 配置/所需权限 profile。
5. 当前部署配置默认。

只有用户选择 “Auto” 时才允许基于健康、质量、成本或延迟推荐；推荐首阶段仍是确定性本地规则，不调用路由 LLM。Direct Lane 与 Auto router 只能消费冻结的 catalog/discovery snapshot；不得在请求热路径同步 probe、update 或创建 capability session。缓存缺失时，显式 Backend 直接尝试目标 driver 并返回真实启动错误；Auto 则 fail clearly 或使用带 freshness 的最后快照。

**Required output:**

```python
@dataclass(frozen=True)
class BackendSelection:
    binding_id: str
    model_id: str | None
    reason_code: str
    considered: tuple[str, ...]
    degraded: bool
```

**Tests first:**

```python
def test_explicit_backend_always_wins_over_auto_policy(): ...
def test_router_rejects_missing_required_capability(): ...
def test_auto_route_is_deterministic_for_same_catalog_snapshot(): ...
def test_route_explanation_contains_no_secret_or_prompt_content(): ...
```

**Acceptance:**

- 不存在跨 provider 的静默 model fallback。
- route reason 可显示在诊断中并进入 audit。

**Verify:**

```bash
uv run pytest tests/test_backend_routing_policy.py tests/contracts/test_direct_programming_lane.py -q
```

**Commit:** `feat(agent-session): add deterministic capability routing`

### Task A7：在静态 Catalog 稳定后增加受信扩展边界

**Files:**

- Create: `src/agent_session/manifest.py`
- Create: `src/agent_session/external_driver.py`
- Create: `schemas/backend-manifest-v1.json`
- Create: `tests/security/test_backend_manifest.py`
- Create: `docs/backend-extension-security.md`

**Implementation:**

1. manifest 只允许声明 descriptor、命令模板参数、capability 和隔离要求。
2. driver 必须在受信 driver registry 中；未知 Python entry point 默认拒绝。
3. 外部命令使用 argv 数组，不接受 shell string。
4. 安装权限与运行权限分离；聊天用户不能安装新 driver。
5. TUI2ACP 的 custom command 必须完整、安全解析 argv；默认 `--unsafe` 行为必须被显式 trust profile 控制。
6. 远程 Agent 适配器只有在本地 conformance kit 通过后再支持，协议特定逻辑封装在 driver 内。

**Tests first:**

```python
def test_unknown_driver_or_manifest_version_fails_closed(): ...
def test_manifest_command_cannot_inject_shell_metacharacters(): ...
def test_chat_user_cannot_install_backend_extension(): ...
def test_external_driver_capability_claims_use_same_conformance_suite(): ...
```

**Acceptance:**

- 插件化不扩大主 Bot 凭据和宿主 Shell 的隐式信任边界。
- manifest 解析失败、未知 capability 或未满足隔离要求均 fail closed。

**Verify:**

```bash
uv run pytest tests/security/test_backend_manifest.py tests/contracts/test_builtin_backend_conformance.py -q
```

**Commit:** `feat(agent-session): add trusted backend manifests`

### Checkpoint CP-A 验证

```bash
uv run pytest tests/test_backend_*.py tests/test_session_*.py tests/contracts/test_*backend* -q
uv run pytest tests/contracts/test_direct_programming_lane.py -q
uv run pytest tests/contracts/test_protected_execution_lanes.py -q
uv run pytest tests/test_deep_engine.py tests/test_deep_completion_guard.py -q
uv run pytest tests/test_spec_engine.py tests/test_spec_task_persistence.py tests/test_adaptive_review_pipeline.py -q
uv run pytest tests/test_acp_*.py tests/test_ttadk.py tests/test_ttadk_*.py -q
uv run pytest tests/ -q -m "not slow"
uv run ruff check src/agent_session src/acp src/ttadk src/feishu src/agent src/mode src/config src/autonomous/team src/worktree_engine src/workflow_engine
```

Direct 的“一次 prompt”按 logical user prompt 计数；provider-local startup/retry 若属于既有成熟合同可有多个 attempt，但不得增加 planner/reviewer/coordinator prompt，且所有 attempt 必须归属于同一个 logical prompt。

手工审计：

```bash
rg -n 'startswith\\(\"ttadk_\"|agent_type\\s*==|_KNOWN_TOOLS|DEFAULT_TOOLS' \
  src/feishu src/agent_session src/acp src/deep_engine src/spec_engine \
  src/worktree_engine src/workflow_engine src/slock_engine
```

每个剩余分支必须位于对应 driver/compatibility shim，或附有无法类型化的具体原因。

---

## 10. Phase B — Durable Orchestration Core

**目标：** 让 Workflow、Team、Slock 和 Worktree 共享可恢复、可验证、可停止的编排事实，同时只通过 adapter 观察 Direct/Deep/Spec。

**预计投入：** 45–70 engineer-days。Phase B 先过“零派发编译检查”，再接入耐久真实执行；这是工程依赖，不是线上灰度周期。

**最短可体验路径：** B1–B6 + B10 + B9 的真实 synthesis 完成后即可让当前 Owner 使用 Workflow IR v2。B7（Team/Slock 收敛）、B8（Worktree 节点）和 B9 的动态 `PlanPatch` 是可独立完成并立即开放的扩展，不阻塞静态有界 DAG 的首版体验。

### Task B1：实现最小 Run Read/Control Contract

**Files:**

- Create/complete: `src/tasking/models.py`
- Create/complete: `src/tasking/protocols.py`
- Create/complete: `src/tasking/control_plane.py`
- Create: `src/tasking/adapters/direct.py`
- Create: `src/tasking/adapters/deep.py`
- Create: `src/tasking/adapters/spec.py`
- Create: `src/tasking/adapters/workflow.py`
- Create: `src/tasking/adapters/worktree.py`
- Create: `src/tasking/adapters/team.py`
- Create: `src/tasking/adapters/slock.py`
- Test: `tests/test_task_control_plane.py`
- Test: `tests/contracts/test_protected_lane_adapters.py`

如果既有收敛计划 Task 13 已创建这些模块，扩展现有实现，不新建第二套 `tasking_v2`。

**Required minimum contract:**

```python
class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

class CancellationState(StrEnum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    ACKNOWLEDGED = "acknowledged"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"

@dataclass(frozen=True)
class RunView:
    run_id: str
    tenant_key: str
    source: str
    strategy: str
    owner_id: str
    project_id: str | None
    thread_id: str | None
    authority_ref: str
    state: RunState
    cancellation: CancellationState
    created_at: datetime
    updated_at: datetime
    active_summary: str
    recoverable: bool

@dataclass(frozen=True)
class ControlResult:
    run_id: str
    tenant_key: str
    cancellation: CancellationState
    reason_code: str

class RunController(Protocol):
    def get(self, run_id: str, actor: Principal) -> RunView | None: ...
    def list(self, query: RunQuery, actor: Principal) -> tuple[RunView, ...]: ...
    async def cancel(self, run_id: str, actor: Principal) -> ControlResult: ...
```

**Implementation:**

1. RunView 是投影，不是第二事实源。
2. Direct/Deep/Spec adapter 只调用现有 manager/session 的公开控制方法。
3. 适配器不得把内部状态推测成更强承诺；Deep 若不能 crash-safe recovery，就返回 `recoverable=False`。
4. 一个 run 只有一个 owner adapter，避免 cancel 双发。
5. 用稳定 namespace 生成 run id；重启后同一耐久 run 不改变 id。
6. 所有 get/list/cancel 先按 tenant、owner/project scope 和 authority 检查；仅凭全局 run id 不可读取或控制。
7. B1 只统一“请求取消”和取消状态。只有实际资源 owner 返回 acknowledgment 才显示已停止；cooperative/未知结果不得包装成成功。

**Tests first:**

```python
def test_run_view_never_claims_recovery_the_engine_does_not_support(): ...
def test_cancel_is_sent_to_exactly_one_owner_adapter(): ...
def test_deep_and_spec_adapters_do_not_change_engine_state_machine(): ...
def test_direct_adapter_adds_no_prompt_or_planner_call(): ...
def test_cross_tenant_run_lookup_and_cancel_are_denied(): ...
def test_cancel_request_is_not_rendered_as_acknowledged_stop(): ...
```

**Acceptance:**

- `/status`、`/stop` 类产品能力可以从一个 tenant-scoped 接口读取或请求控制所有通道，并诚实显示 requested/acknowledged/unknown/unsupported。
- 关闭 Task Control Plane 不影响原 direct/deep/spec 入口。

**Verify:**

```bash
uv run pytest tests/test_task_control_plane.py tests/contracts/test_protected_lane_adapters.py -q
```

**Commit:** `feat(tasking): add execution-lane control adapters`

### Task B2：定义冻结、版本化的 Orchestration IR

**Files:**

- Create: `src/orchestration/__init__.py`
- Create: `src/orchestration/domain.py`
- Create: `src/orchestration/validation.py`
- Create: `src/orchestration/serialization.py`
- Modify: `src/autonomous/domain/plans.py`
- Modify: `src/autonomous/domain/effects.py`
- Create: `src/orchestration/compiler.py`
- Test: `tests/orchestration/test_domain.py`
- Test: `tests/orchestration/test_validation.py`
- Test: `tests/orchestration/test_durable_plan_compiler.py`
- Create: `docs/orchestration-ir-v1.md`

**Required types:**

```python
class NodeKind(StrEnum):
    AGENT = "agent"
    DECISION = "decision"
    MAP = "map"
    LOOP = "loop"
    SYNTHESIS = "synthesis"
    HUMAN_APPROVAL = "human_approval"
    WORKTREE = "worktree"
    SUBRUN = "subrun"

class EffectVisibility(StrEnum):
    NONE = "none"
    MEDIATED = "mediated"
    OPAQUE_SESSION = "opaque_session"

@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    backoff_seconds: tuple[float, ...]
    retryable_error_codes: frozenset[str]

@dataclass(frozen=True)
class NodeBudget:
    wall_seconds: int
    max_agent_calls: int
    max_output_tokens: int | None

@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    kind: NodeKind
    role: str
    dependencies: tuple[str, ...]
    inputs: tuple["ArtifactSelector", ...]
    outputs: tuple["ArtifactContract", ...]
    backend_requirement: "BackendRequirement | None"
    retry: RetryPolicy
    budget: NodeBudget
    done_criteria: tuple[str, ...]
    risk_class: "RiskClass"
    effect_visibility: EffectVisibility

@dataclass(frozen=True)
class OrchestrationPlan:
    schema_version: int
    plan_id: str
    revision: int
    goal: str
    nodes: tuple[NodeSpec, ...]
    max_total_agent_calls: int
    max_total_output_tokens: int | None
    max_total_nodes: int
    max_plan_revisions: int
    deadline_at: datetime
```

**Canonical boundary:**

`OrchestrationPlan` 是用户/LLM/模板产生的**输入语言**，不是第二套耐久状态机。Admission 必须把一个已验证 IR revision 原子编译为仓库现有 frozen `src/autonomous/domain/plans.py::{Plan, PlanStep}`（若现有字段不足，版本化演进这些 canonical 类型），随后只有 canonical aggregate/event 是运行事实。IR node id、durable step id 和 revision 的映射稳定写入编译事件；`Attempt` 只在 scheduler 实际排队时确定性创建，`Effect` 只在真实调用/mediated tool action 准备派发时创建，不能在编译期预造。Team/Workflow adapter 不能让 IR projection 与 Autonomous domain 各自拥有终态。

**Validation rules:**

- 普通依赖必须构成 DAG。
- `loop` 是显式节点，必须有最大轮数、预算和结构化停止条件。
- `map` 必须有 fanout 上限。
- 所有输入引用必须由先行节点或 plan input 提供。
- 每个外部写节点必须有 risk class 和 policy hook。
- backend requirement 只声明能力，不硬编码员工身份；显式 Run binding 可覆盖。
- agent-call/token 上限按所有可达节点、retry、map、loop 的**最坏有界展开**累计；wall time 按并行 DAG critical path、retry deadline 和总 deadline 校验，不能用“最短路径预算”代替。
- `max_total_nodes`、`max_plan_revisions`、map fanout 和 loop iterations 均不可缺省。
- `MEDIATED` 只用于 driver 能逐工具上报并门控 Effect 的路径。ACP/CLI 内部副作用不可见时，整个 Agent session 标为 `OPAQUE_SESSION` 高风险 Effect；只有在可证明的 OS/filesystem/network sandbox 或明确人工批准下运行。普通 Git worktree 只隔离代码分支，不隔离宿主文件系统、进程或网络，不能满足安全 isolation gate。
- 可能产生副作用的 `OPAQUE_SESSION` 强制 `max_attempts=1`；dispatch 后的 timeout、cancel、断连或崩溃一律进入 `effect.unknown` 并禁止自动 retry，等待 reconcile/人工处置。只有已证明 side-effect-free 的 opaque 分析 session 才能按普通 retry policy 重试。

**Tests first:**

```python
def test_cycle_is_rejected_outside_bounded_loop_node(): ...
def test_unbounded_map_or_loop_is_rejected(): ...
def test_missing_artifact_producer_is_rejected(): ...
def test_external_write_without_risk_class_is_rejected(): ...
def test_round_trip_preserves_frozen_plan_hash(): ...
def test_ir_compiles_atomically_to_one_canonical_durable_plan(): ...
def test_ir_and_durable_plan_step_ids_and_revisions_remain_stable(): ...
def test_attempt_and_effect_ids_are_created_only_at_runtime_boundaries(): ...
def test_worst_case_retry_map_loop_budget_is_enforced(): ...
def test_opaque_side_effectful_session_requires_isolation_or_approval(): ...
def test_opaque_side_effectful_session_never_auto_retries_after_dispatch_unknown(): ...
def test_plain_git_worktree_does_not_satisfy_os_sandbox_capability(): ...
```

**Acceptance:**

- IR 仅描述可验证的执行意图，不包含任意 Python/JS 代码。
- schema version 未知时 fail closed。
- 一个 logical run 只有 canonical durable aggregate 的一个 terminal truth；IR 不能独立进入终态。

**Verify:**

```bash
uv run pytest tests/orchestration/test_domain.py tests/orchestration/test_validation.py tests/orchestration/test_durable_plan_compiler.py -q
```

**Commit:** `feat(orchestration): define versioned plan IR`

### Task B3：建立 Artifact、Provenance 与 Done Criteria 合同

**Files:**

- Create: `src/orchestration/artifacts.py`
- Create: `src/orchestration/criteria.py`
- Create: `src/orchestration/artifact_store.py`
- Test: `tests/orchestration/test_artifacts.py`
- Test: `tests/orchestration/test_done_criteria.py`

**Required artifact metadata:**

```python
@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    tenant_key: str
    project_id: str | None
    authority_ref: str
    media_type: str
    schema_id: str | None
    content_hash: str
    blob_ref: str
    producer_run_id: str
    producer_node_id: str
    producer_attempt_id: str
    source_artifacts: tuple[str, ...]
    sensitivity: "SensitivityClass"
    acl_ref: str
    created_at: datetime
```

**Implementation:**

1. 大内容进入现有加密 Blob/DataService；Journal 只保存引用和 hash。
2. 文本结果也必须具有 provenance，禁止把缓存输出当作无来源的新结果。
3. code patch、test report、review verdict、decision、user approval 定义首批 schema。
4. done criteria 分为 deterministic、reviewer、human approval；不得仅凭“有文本输出”完成。
5. Artifact 删除/保留遵循项目、员工、群上下文的 TTL 和数据删除合同。
6. 所有 read/list/resolve 都接收 Principal 并验证 tenant/project/ACL/sensitivity；不可通过猜测 artifact id 跨租户读取。

**Tests first:**

```python
def test_artifact_hash_and_lineage_survive_replay(): ...
def test_cached_artifact_cannot_hide_changed_input_provenance(): ...
def test_missing_required_test_report_blocks_done_criteria(): ...
def test_sensitive_artifact_uses_encrypted_blob_store(): ...
def test_cross_tenant_artifact_read_is_denied(): ...
```

**Acceptance:**

- Agent 间交接使用 ArtifactRef，不复制无边界的全量对话。
- 最终摘要可追溯到输入、执行 Agent、model 和验证证据。

**Verify:**

```bash
uv run pytest tests/orchestration/test_artifacts.py tests/orchestration/test_done_criteria.py -q
```

**Commit:** `feat(orchestration): add artifact and completion contracts`

### Task B4：为 Workflow 建立 IR v2 编译、诊断与直接启用路径

**Files:**

- Create: `src/workflow_engine/ir_adapter.py`
- Create: `src/workflow_engine/ir_script_compiler.py`
- Create: `src/workflow_engine/observed_trace.py`
- Modify: `src/workflow_engine/script_gen.py`
- Modify: `src/workflow_engine/runtime/runtime.js`
- Modify: `src/workflow_engine/templates.py`
- Create: `tests/test_workflow_ir_observation.py`
- Create: `tests/test_workflow_ir_compiler.py`

**Migration rule:**

现有 JS Workflow v1 继续作为兼容和紧急回退路径；IR v2 只有在编译检查、B5/B6/B10 的耐久执行合同与 B9a 真实综合完成后才接管对应请求。两条路径不按租户分流。

**Implementation:**

1. **B4a 离线/诊断 observation：** runtime instrumentation 把 v1 已经发生的 host call 纯确定性转换为 `ObservedOrchestrationTrace`。它只用于测试和人工诊断 backend/model、role、依赖、调用数、预算和终态漂移，不形成线上观察期，也不产生第二次 LLM/Agent 调用。
2. **B4b v2 dry-run：** v2 请求由一次 generator 直接输出 schema-valid JSON IR，不同时生成 v1 script；deterministic compiler 完成 validate/compile/dry-run，且此步骤不得派发 worker Agent。
3. CP-B-Compile 通过后保留 v2 的编译入口；B5/B6/B9a/B10 和 CP-B-Execute 通过后，v2 在当前 Owner 的正常 Workflow 入口直接可选，实际执行仍只以一次 IR generator 输出为源。
4. `WORKFLOW_IR_V2_ENABLED` 只作为紧急回退到 v1 的总开关，不接受 tenant、百分比或 maturity 参数。
5. 不可表达的 v1 动态 JS 明确标记 `unsupported_in_ir_v1`，继续走 v1；禁止静默降级语义。
6. 逐个迁移 built-in templates，先 sequence/fanout/verify，再 decision/map/loop。

**Tests first:**

```python
def test_ir_compiler_preserves_backend_model_and_reviewer_bindings(): ...
def test_unsupported_dynamic_js_stays_on_v1_with_explicit_reason(): ...
def test_observation_diagnostic_adds_no_generator_or_worker_agent_call(): ...
def test_v2_uses_one_ir_generation_source_and_no_parallel_v1_generation(): ...
def test_builtin_template_ir_has_same_declared_call_budget(): ...
def test_remote_call_ledger_counts_generation_and_execution_attempts(): ...
def test_v2_switch_is_global_rollback_not_tenant_rollout(): ...
```

**Acceptance:**

- Observation diagnostic 不增加第二次 generator 或 worker Agent 调用。
- Observed trace 与 v2 IR 是不同类型；二者差异被记录，不能被当作成功或等价证明。
- CP-B-Execute 与 B9a 通过后，Owner 无需 allowlist 或等待观察期即可选择 v2；关闭总开关可立即回到 v1。

**Verify:**

```bash
uv run pytest tests/test_workflow_ir_observation.py tests/test_workflow_ir_compiler.py -q
node --test src/workflow_engine/runtime/*.test.js
```

**Commit:** `feat(workflow): add validated IR v2 path`

### Task B5：将耐久 Orchestration Event 接到现役生产端口

**Files:**

- Create: `src/orchestration/events.py`
- Create: `src/orchestration/ports.py`
- Create: `src/orchestration/projection.py`
- Create: `src/orchestration/dispatch.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/autonomous/gateway/coordinator.py`
- Modify: `src/autonomous/broker/dispatch_gate.py`
- Modify: `src/autonomous/policy/policy_engine.py`
- Modify: `src/workflow_engine/journal.py`
- Test: `tests/orchestration/test_projection_replay.py`
- Test: `tests/orchestration/test_effect_dispatch.py`

**Event sequence:**

```text
plan.accepted
node.ready
attempt.queued
attempt.prepared
attempt.executing
effect.prepared
effect.executing
effect.committed | effect.unknown | effect.rejected
attempt.succeeded | attempt.failed | attempt.cancelled
node.succeeded | node.failed | node.cancelled
run.succeeded | run.failed | run.cancelled
```

**Implementation:**

1. `src/orchestration` 只依赖新的 `JournalPort`、`PolicyPort`、`EffectDispatchPort`，不得直接绑定 legacy concrete class。
2. 先用 conformance test 审计三条现实：production `EmployeeDepartmentRuntime` 当前使用 `EmployeeDispatchCoordinator`；旧 `DispatchGate` 在 `effect.executing` 后未再次 anchor；旧 `PolicyEngine` 的 approval/standing order 仍以内存为主。它们在修复并接入现役 composition 前都不能被宣称为生产安全内核。
3. 以现役 `JournalWriter` 为唯一写入核心，实现/适配生产 ports；严格顺序为 policy decision → `effect.prepared` fsync+anchor → `effect.executing` fsync+anchor → adapter invocation。缺少 EXECUTING 时不得调用外部 adapter。
4. 逐类迁移 live Employee/Team/Workflow Effect；迁移期间一个 Effect kind 只有一个 dispatch owner，禁止 legacy gate 与新 port 双派发。
5. run 不能在 unresolved Effect 存在时进入终态。
6. 对 `OPAQUE_SESSION` 只锚定整个 session dispatch/结果，不能声称观察到内部工具 Effect；MEDIATED driver 才能产生逐工具 Effect。
7. `src/workflow_engine/journal.py` 若继续只做结果缓存，应重命名/别名为 `WorkflowInvocationCache`，避免与 lifecycle Journal 混淆。
8. cache key 加入输入 Artifact hash、backend/model/capability version、code/template version 和 side-effect-free 标志；有副作用节点默认不缓存。

**Tests first:**

```python
def test_replay_rebuilds_identical_run_projection(): ...
def test_external_call_never_starts_before_prepared_frame_is_anchored(): ...
def test_executing_frame_is_anchored_before_adapter_invocation(): ...
def test_crash_after_prepared_before_executing_is_not_treated_as_dispatched(): ...
def test_terminal_run_rejects_unresolved_effect(): ...
def test_side_effectful_node_is_never_reused_from_invocation_cache(): ...
def test_legacy_and_new_dispatch_owner_never_send_same_effect(): ...
def test_opaque_session_does_not_claim_nested_effect_exactly_once(): ...
```

**Acceptance:**

- Journal 是 lifecycle SSOT；cache 只是可丢失优化。
- kill -9 后能够解释每个 attempt 是未开始、已提交还是 unknown。
- 只有通过 production port conformance 并在 `EmployeeDepartmentRuntime` 组装的实现可称为现役；legacy importable class 不构成证据。

**Verify:**

```bash
uv run pytest tests/orchestration/test_projection_replay.py tests/orchestration/test_effect_dispatch.py -q
uv run pytest tests/autonomous/unit/test_journal* tests/autonomous/chaos/ -q
```

**Commit:** `feat(orchestration): anchor run and effect lifecycle`

### Task B6：实现拥有资源生命周期的 Scheduler、Budget 与 CancellationScope

**Files:**

- Create: `src/orchestration/scheduler.py`（仅定义 `RunSchedulerPort`）
- Create: `src/orchestration/budget.py`
- Create: `src/orchestration/cancellation.py`
- Modify: `src/autonomous/scheduler/scheduler.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/tasking/scheduler.py`
- Modify: `src/feishu/control_plane.py`
- Modify: `src/workflow_engine/bridge.py`
- Modify: `src/workflow_engine/executor.py`
- Modify: `src/worktree_engine/dispatcher.py`
- Test: `tests/orchestration/test_scheduler.py`
- Test: `tests/orchestration/test_budget.py`
- Test: `tests/orchestration/test_cancellation.py`
- Test: `tests/test_workflow_hard_deadline.py`

**Implementation:**

1. 先固化 scheduler ownership：
   - `src/tasking/scheduler.py` 保留普通 foreground/保护路径的既有进程内资源调度，不接管耐久 IR。
   - `src/feishu/control_plane.py` 只负责命令 gate/deferred exit，不 admission 或 dispatch run。
   - `src/orchestration/scheduler.py` 只定义 port、budget/cancel 协议，不持有第二个队列。
   - 版本化迁移 `src/autonomous/scheduler/scheduler.py` 为新 IR/Team/Proactive 的唯一耐久 run scheduler。
   - Engine worker pool 只拥有已分配 attempt 的本地资源，不重新 admission。
2. 当前 Autonomous scheduler 的 queue、lease、fencing counter 以内存为主；先迁移为 Journal event + replay projection，持久保存 scheduler owner epoch/lease/fencing。旧进程与新进程重叠时，较旧 epoch 的 admission/dispatch/commit 全部拒绝。
3. 在 `EmployeeDepartmentRuntime` production composition 中实际组装、恢复和关闭唯一 Durable scheduler；没有 wiring test 不能宣称上线。
4. Durable scheduler 管理当前 Owner 的全局/project/backend 并发与简单有界队列；首版不建设多租户公平调度、复杂权重或防饥饿策略。
5. 每个 run/node/attempt 有绝对 deadline；in-flight 调用不能无限延长总 deadline。
6. `CancellationScope` 持有实际 session/process handle，执行 cancel → bounded join → acknowledged/unknown。
7. budget 至少覆盖 agent calls、wall time、output tokens；若 provider 提供 usage，再覆盖 token/cost。
8. budget exhausted 进入明确终态或等待人工扩容，不得自动提高。
9. race loser、timeout、用户 stop 和全局 kill 走同一取消协议。

**Tests first:**

```python
def test_deadline_does_not_extend_for_inflight_agent(): ...
def test_cancel_acknowledges_every_owned_session_or_marks_unknown(): ...
def test_race_loser_cannot_commit_late_result(): ...
def test_budget_exhaustion_blocks_new_attempt_without_losing_checkpoint(): ...
def test_scheduler_enforces_owner_project_and_backend_concurrency_limits(): ...
def test_old_and_durable_scheduler_never_both_dispatch_same_run(): ...
def test_feishu_control_plane_never_becomes_run_dispatch_owner(): ...
def test_scheduler_replays_queue_lease_and_fencing_epoch_after_restart(): ...
def test_old_scheduler_process_cannot_dispatch_or_commit_after_new_owner_epoch(): ...
def test_employee_department_runtime_composes_recovers_and_stops_durable_scheduler(): ...
```

**Acceptance:**

- “停止成功”意味着资源已确认停止；否则必须显示“停止状态未知”。
- 所有硬预算可从 RunView 和卡片查看。

**Verify:**

```bash
uv run pytest tests/orchestration/test_scheduler.py tests/orchestration/test_budget.py tests/orchestration/test_cancellation.py tests/test_workflow_hard_deadline.py -q
```

**Commit:** `feat(orchestration): enforce owned cancellation and budgets`

### Task B7：明确 Team 与 Slock 的职责并收敛事实源

**Files:**

- Create: `src/orchestration/adapters/team.py`
- Create: `src/orchestration/adapters/slock.py`
- Modify: `src/autonomous/team/coordinator.py`
- Modify: `src/autonomous/team/models.py`
- Modify: `src/slock_engine/task_queue.py`
- Modify: `src/slock_engine/task_board_manager.py`
- Modify: `src/slock_engine/engine.py`
- Create: `tests/orchestration/test_team_slock_ownership.py`
- Create: `tests/orchestration/test_slock_queue_migration.py`

**Ownership decision:**

- Slock owns：群消息分类、activation、参与者选择、交互卡、兼容入口。
- Team v2 owns：多 Employee run、assignment、handoff、review/revise/finalize。
- Canonical durable domain owns：run/step/attempt/effect/artifact/control 语义与唯一终态。
- Journal owns：唯一耐久事件事实；projection 只可从 Journal 重建。

**Implementation:**

1. Slock 接收到已授权复杂任务后创建一个稳定 team/orchestration run，不再复制执行任务到第二个内存队列。
2. 迁移采用：停止旧队列接收 → 快照可恢复项 → 生成稳定 Run ID → Journal 单写 → 核对投影 → 关闭旧消费者。
3. 禁止长期双写；迁移期的对账写入也不能有两个 dispatcher。
4. `slock_default_roles` 为空时作为“无预设角色”信息状态，不是启动错误；需要角色的任务在执行前明确提示并阻断，而不是启动时制造误报。
5. Team Coordinator 产出的 assignment 和 handoff 进入同一 projection。

**Tests first:**

```python
def test_slock_classification_creates_exactly_one_team_run(): ...
def test_old_queue_and_new_runtime_never_both_dispatch_same_task(): ...
def test_empty_default_roles_allows_startup_but_blocks_role_required_run(): ...
def test_team_assignment_is_visible_through_shared_run_view(): ...
```

**Acceptance:**

- 重启不丢 Slock 等待任务。
- 一个业务任务只有一个执行 owner 和一个 terminal truth。
- 本任务测试通过后，Team/Slock 耐久执行直接对 Owner 可用，不等待其他 extension。

**Verify:**

```bash
uv run pytest tests/orchestration/test_team_slock_ownership.py tests/orchestration/test_slock_queue_migration.py -q
uv run pytest tests/test_slock_*.py tests/autonomous/unit/test_team* -q
```

**Commit:** `refactor(collaboration): unify slock and team run ownership`

### Task B8：将 Worktree 作为受控黑盒节点接入

**Files:**

- Create: `src/orchestration/adapters/worktree.py`
- Modify: `src/worktree_engine/manager.py`
- Modify: `src/worktree_engine/models.py`
- Test: `tests/orchestration/test_worktree_node.py`

**Implementation:**

1. `WORKTREE` node 输入是 goal、repo snapshot、unit plan、merge policy 和 budget。
2. 输出是 branch refs、patch Artifact、test Artifact、review verdict 和 conflict disclosure。
3. Orchestration Runtime 只调用 Worktree manager 的公开 run/cancel/status 接口，不操作内部 future。
4. Worktree node 只有在 Task 0.5 的 terminal truth gate 通过后才能用于真实执行。
5. merge/push 等外部 Effect 单独锚定并走 policy，不因为 node succeeded 自动执行。

**Tests first:**

```python
def test_worktree_node_exposes_patch_test_review_artifacts(): ...
def test_failed_worktree_unit_fails_node_and_blocks_merge_effect(): ...
def test_orchestration_cancel_delegates_to_worktree_and_waits_for_ack(): ...
```

**Acceptance:**

- Worktree 保持专业隔离策略，不被拆成通用 scheduler 内部细节。
- 编排器看见真实 unit 和 merge 状态。
- 本任务测试通过后，Worktree 节点直接对 Owner 可用，不等待 Team/Slock 或动态 PlanPatch。

**Verify:**

```bash
uv run pytest tests/orchestration/test_worktree_node.py tests/test_worktree_*.py -q
```

**Commit:** `feat(orchestration): add bounded worktree node adapter`

### Task B9：先实现真实综合，再扩展有界 PlanPatch

#### B9a：真实 Reviewer 与 Synthesis（Owner 首版路径）

**Files:**

- Create: `src/orchestration/synthesis.py`
- Test: `tests/orchestration/test_synthesis.py`

**Implementation:**

1. synthesis 必须消费所有 required Artifact，披露冲突和缺失证据。
2. Reviewer 是真实节点/attempt；提示中出现“请审查”不构成 review 证据。
3. B9a 完成即可随静态有界 DAG 向 Owner 开放，不等待动态重规划。

**Tests first:**

```python
def test_synthesis_reports_conflicting_required_artifacts(): ...
def test_review_claim_requires_real_reviewer_attempt(): ...
def test_synthesis_cannot_ignore_missing_required_artifact(): ...
```

**Acceptance:**

- 静态 DAG 能产出可追溯、披露冲突且有真实 Reviewer 的最终结果。

**Verify:**

```bash
uv run pytest tests/orchestration/test_synthesis.py -q
```

**Commit:** `feat(orchestration): require real review and synthesis`

#### B9b：有界 PlanPatch 与动态协调（独立扩展）

**Files:**

- Create: `src/orchestration/patches.py`
- Create: `src/orchestration/coordinator.py`
- Modify: `src/autonomous/team/coordinator.py`
- Test: `tests/orchestration/test_plan_patch.py`
- Test: `tests/orchestration/test_dynamic_coordination.py`

**Implementation:**

1. Coordinator 不能直接修改 plan；只能提议 `PlanPatch`。
2. patch 包含稳定 patch id、base revision、add/replace/cancel node、理由、预算变化和 done criteria 变化。
3. validator 重新检查 DAG、预算、权限和 fanout；通过后写 `plan.revised`。
4. 默认不得降低原 done criteria 或风险等级；降低必须人工批准。
5. 每个 plan 强制 `max_plan_revisions`、`max_total_nodes`、`max_patch_rate` 和 coordinator wall/token budget；重复 patch id 幂等，patch storm 直接暂停并升级。

**Tests first:**

```python
def test_stale_plan_patch_is_rejected_by_revision(): ...
def test_coordinator_cannot_raise_budget_or_lower_criteria_without_approval(): ...
def test_duplicate_patch_is_idempotent_and_revision_ceiling_stops_patch_storm(): ...
```

**Acceptance:**

- 动态性存在，但永远受 revision、预算、权限和停止条件限制。
- 每次重规划均可审计和回放。
- B9b 未完成时系统仍可诚实运行静态有界 DAG，不阻塞 B9a 的真实综合能力。
- B9b 测试通过后动态协调直接成为 Owner 可选能力。

**Verify:**

```bash
uv run pytest tests/orchestration/test_plan_patch.py tests/orchestration/test_dynamic_coordination.py -q
```

**Commit:** `feat(orchestration): add bounded dynamic plan revision`

### Task B10：恢复、卡片和运维视图

**Files:**

- Create: `src/orchestration/recovery.py`
- Create: `src/orchestration/migrations.py`
- Create: `src/orchestration/diagnostics.py`
- Create: `src/orchestration/renderer.py`
- Modify: `src/card/session/core.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Create: `tests/orchestration/test_restart_reconcile.py`
- Create: `tests/orchestration/test_schema_migration.py`
- Create: `tests/orchestration/test_progress_projection.py`
- Create: `ux/orchestration-run-card.html`

**Implementation:**

1. startup replay 后逐个处理 queued、executing、unknown effect 和 cancellation-pending。
2. 不自动重跑 unknown 外部 Effect；先 reconcile 或人工处置。
3. 进度卡展示 goal、active nodes、blocked reason、backend/model、budget、latest artifact、cancel acknowledgment 和 recovery state。
4. 修复 Workflow snapshot 丢失 `current_activity` 一类“内部更新但卡片不可见”的投影断裂。
5. 保留普通编程主卡和现有折叠执行记录，不将 Orchestration UI 强加给 Direct Lane。
6. UI 实施前先更新 `ux/` 预览并经审查。
7. 在任何新 Journal event 上线前实现 old snapshot/event → new projection migration、重复迁移幂等和未知未来 schema fail-closed；代码回滚不认识新 schema 时必须拒绝启动，不能丢帧。

**Tests first:**

```python
def test_restart_replays_only_safe_uncommitted_attempts(): ...
def test_unknown_effect_requires_reconciliation_before_terminal(): ...
def test_progress_snapshot_preserves_current_activity_and_budget(): ...
def test_direct_programming_card_has_no_orchestration_sections(): ...
def test_old_snapshot_and_events_migrate_idempotently(): ...
def test_unknown_future_schema_blocks_startup_without_mutation(): ...
```

**Acceptance:**

- kill/restart 后 run 的状态、活动说明和预算不倒退。
- 卡片不把 cache hit、review 建议或 queued 状态误报为已完成。

**Verify:**

```bash
uv run pytest tests/orchestration/test_restart_reconcile.py tests/orchestration/test_progress_projection.py tests/orchestration/test_schema_migration.py -q
uv run pytest tests/test_card_*.py tests/test_workflow_renderer.py -q
```

**Commit:** `feat(orchestration): add recovery and truthful run views`

### Checkpoint CP-B-Compile：零派发编译

- IR v2 完成 schema validate、deterministic compile 和 dry-run；这一阶段不得派发 worker 或外部 Effect。
- v1 observation/comparator 只作为自动化或人工诊断工具，对实际执行过且可观察的调用子集比较 call ledger，不把 trace 与 IR 宣称为完整语义等价。
- 已实现的 Team/Slock adapter 在此阶段只能提供固定命名空间的只读投影，但它们不是 Workflow IR 编译检查的前置。
- Direct/Deep/Spec 所有保护合同通过。
- 没有额外 Agent 调用、外部 Effect 或重复卡片。

### Checkpoint CP-B-Execute：耐久真实执行

**Executable chaos evidence:**

- Create: `tests/orchestration/chaos_manifest.json`
- Create: `scripts/run_orchestration_chaos.py`
- Create: `tests/orchestration/test_chaos_manifest.py`

Manifest 中每个 case 必须给出 case id、注入帧、最大墙钟、预期 projection/Effect/资源状态和测试输出路径；不能只在文档中写一句“做过 kill 测试”。

```bash
uv run pytest \
  tests/orchestration/test_projection_replay.py \
  tests/orchestration/test_effect_dispatch.py \
  tests/orchestration/test_scheduler.py \
  tests/orchestration/test_budget.py \
  tests/orchestration/test_cancellation.py \
  tests/orchestration/test_synthesis.py \
  tests/orchestration/test_restart_reconcile.py \
  tests/orchestration/test_schema_migration.py \
  tests/orchestration/test_progress_projection.py -q
uv run pytest tests/autonomous/unit/test_journal* tests/autonomous/chaos/test_journal* -q
uv run pytest tests/test_workflow_*.py -q
uv run pytest tests/ -q -m "not slow"
uv run ruff check src/orchestration src/tasking src/autonomous src/workflow_engine
uv run python scripts/run_orchestration_chaos.py --manifest tests/orchestration/chaos_manifest.json
```

额外故障注入：

- 在 PREPARED 后、外部调用前 kill。
- 在远端成功、本地 COMMITTED 前 kill。
- cancel 时 provider 不响应。
- Journal anchor 失败、磁盘满、Blob 缺失。
- Workflow Node 进程与 durable scheduler 重启。
- 重复 callback、乱序 event。
- old→new migration、重复 migration、new→old 不兼容版本 fail-closed。

每个故障项都必须产生可复现测试报告。只有 projection 完整、无重复 session dispatch/mediated Effect、所有 opaque 未知状态均披露且 schema 回滚检查通过，才允许 IR 驱动真实路径；不可见的 nested side effect 不作无法证明的“零重复”承诺。CP-B-Execute 与 B9a 检查通过后，Workflow IR v2 立即对当前 Owner 可用。

**Extension checks：** B7 的 Team/Slock 双 owner、迁移恢复和 Channel 重启；B8 的 Worktree cancel/terminal/merge Effect；B9b 的 stale/duplicate `PlanPatch` 与 patch storm，分别随对应任务验证并在通过后独立开放，不反向阻塞 Workflow IR v2。

---

## 11. Phase C — Proactive Work Loop

**目标：** 让 Agent 能在明确的 Goal、Trigger、Authority、Budget 和通知边界内主动工作；不允许 Agent 自设无限目标或绕过人工入口。

**预计投入：** 30–45 engineer-days。首版只接 one-shot/interval/cron 与既有飞书事件，不建设多租户管理、Webhook 服务或多级开放状态。

### Task C1：建立耐久 GoalTemplate、TriggerDefinition 和 AuthorityEnvelope

**Files:**

- Modify: `src/autonomous/domain/goals.py`
- Create: `src/autonomous/domain/triggers.py`
- Create: `src/autonomous/domain/authority.py`
- Create: `src/autonomous/proactive/projection.py`
- Create: `src/autonomous/proactive/events.py`
- Test: `tests/autonomous/unit/test_proactive_domain.py`
- Test: `tests/autonomous/unit/test_proactive_projection.py`

不要与现有 `TriggerSubscription` 并列创建第二套同义类型；先编写 migration/compat parser，再收敛到一个 schema。

**Required types:**

```python
class MisfirePolicy(StrEnum):
    SKIP = "skip"
    FIRE_ONCE = "fire_once"
    CATCH_UP_BOUNDED = "catch_up_bounded"

@dataclass(frozen=True)
class BudgetLimit:
    wall_seconds: int
    max_agent_calls: int
    max_output_tokens: int | None
    max_cost_minor_units: int | None

@dataclass(frozen=True)
class DataEgressPolicy:
    allowed_backend_families: tuple[str, ...]
    allowed_model_ids: tuple[str, ...]
    allowed_egress_domains: tuple[str, ...]
    allowed_context_scopes: tuple[str, ...]
    max_sensitivity: "SensitivityClass"

@dataclass(frozen=True)
class AuthorityEnvelope:
    grant_id: str
    issued_by: "PrincipalRef"
    issued_at: datetime
    issuer_acl_revision: int
    owner_id: str
    tenant_key: str
    allowed_project_ids: tuple[str, ...]
    allowed_chat_ids: tuple[str, ...]
    allowed_action_kinds: frozenset["ActionKind"]
    max_risk: "RiskClass"
    max_occurrences: int
    budget: BudgetLimit
    run_deadline_seconds: int
    data_egress: DataEgressPolicy
    policy_version: str
    policy_digest: str
    approval_digest: str
    resource_parameters_digest: str
    expires_at: datetime
    approval_ref: str | None

@dataclass(frozen=True)
class TriggerDefinition:
    trigger_id: str
    revision: int
    goal_template_id: str
    kind: str
    schedule_or_filter: Mapping[str, object]
    authority: AuthorityEnvelope
    misfire_policy: MisfirePolicy
    enabled: bool

@dataclass(frozen=True)
class TriggerOccurrence:
    occurrence_id: str
    trigger_id: str
    trigger_revision: int
    source_event_id: str
    scheduled_for: datetime | None
    observed_at: datetime
    dedupe_key: str
```

**Implementation:**

1. create/pause/resume/expire/revoke 全部写 Journal。
2. authority 不可由 Agent 扩大；PlanPatch 只能缩小。
3. issuer/owner、tenant、project/chat scope、typed action/risk、次数、wall/agent/token/cost budget、run deadline、TTL、policy/resource digest 和 data-egress policy 缺一则 trigger 无法启用。
4. GoalTemplate 引用版本化 IR template 或单 Employee 任务模板，不保存未经清理的完整聊天历史。
5. replay 可重建 enabled trigger 和已消费 occurrence。
6. “只读”仍可能把代码、群消息或员工记忆发送给外部 provider，属于不可逆数据披露；每次 Backend binding/context selection 都必须满足 `DataEgressPolicy`，不能只检查 filesystem write。
7. grant 的签发、启用、撤销都验证 tenant-bound Principal、当时 ACL revision 与 approval digest；自由字符串身份不能授权。
8. 当前部署的 `tenant_key` 从本机配置固定派生，只作事件、授权、Artifact 和恢复命名空间；不建设 tenant 管理、发布 allowlist 或跨租户配置面。只有配置的 Owner principal 能创建、修改、启用、撤销 Trigger。

**Tests first:**

```python
def test_trigger_without_owner_expiry_or_budget_is_rejected(): ...
def test_agent_plan_patch_cannot_expand_authority(): ...
def test_replay_rebuilds_enabled_and_consumed_occurrences(): ...
def test_revoked_authority_blocks_future_occurrences_immediately(): ...
def test_read_only_external_prompt_still_requires_data_egress_authority(): ...
def test_authority_is_bound_to_policy_and_resource_parameter_digests(): ...
def test_unauthorized_or_cross_tenant_principal_cannot_issue_enable_or_revoke_grant(): ...
```

**Acceptance:**

- “主动”是用户授予的有限能力，不是 Agent 的隐式属性。
- trigger 定义和每次发生均可审计。

**Verify:**

```bash
uv run pytest tests/autonomous/unit/test_proactive_domain.py tests/autonomous/unit/test_proactive_projection.py -q
```

**Commit:** `feat(autonomous): add durable proactive domain`

### Task C2：实现可测试的时间、cron 与 misfire 语义

**Files:**

- Create: `src/autonomous/proactive/clock.py`
- Create: `src/autonomous/proactive/schedule.py`
- Modify: `src/autonomous/scheduler/triggers.py`
- Test: `tests/autonomous/unit/test_proactive_schedule.py`
- Test: `tests/autonomous/chaos/test_trigger_clock_anomalies.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Implementation:**

1. 第一小步支持 one-shot UTC 时间和固定 interval。
2. 第二小步通过 `uv add croniter` 增加 cron 解析，并把解析后的版本锁进 `uv.lock`；不自行实现不完整 cron parser。
3. schedule calculation 接收注入的 Clock，不直接散用 `time.time()`。
4. 明确定义 timezone、DST、时钟回拨、长时间停机和 catch-up 上限。
5. occurrence id 基于 `trigger_id + revision + scheduled_for/source_event_id` 稳定生成。
6. 重启扫描只生成策略允许的 misfire，不形成触发风暴。

**Tests first:**

```python
def test_same_schedule_slot_has_same_occurrence_id_after_restart(): ...
def test_clock_rollback_does_not_refire_consumed_slot(): ...
def test_catch_up_is_bounded_after_long_downtime(): ...
def test_dst_gap_and_fold_follow_documented_timezone_policy(): ...
```

**Acceptance:**

- 对相同定义和时钟输入，计算结果确定。
- 任意停机时长不会产生无界补跑。

**Verify:**

```bash
uv run pytest tests/autonomous/unit/test_proactive_schedule.py tests/autonomous/chaos/test_trigger_clock_anomalies.py -q
```

**Commit:** `feat(autonomous): define durable schedule semantics`

### Task C3：实现 Trigger Adapter 和统一 Occurrence Ingress

**Files:**

- Create: `src/autonomous/proactive/adapters/base.py`
- Create: `src/autonomous/proactive/adapters/schedule.py`
- Create: `src/autonomous/proactive/adapters/feishu_event.py`
- Create: `src/autonomous/proactive/ingress.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/autonomous/ingress/router.py`
- Test: `tests/autonomous/integration/test_trigger_adapters.py`
- Test: `tests/autonomous/security/test_trigger_authentication.py`

**Adapter contract:**

```python
class TriggerAdapter(Protocol):
    async def observe(self) -> AsyncIterator[RawTriggerEvent]: ...
    def authenticate(self, event: RawTriggerEvent) -> SourcePrincipal: ...
    def normalize(self, event: RawTriggerEvent) -> TriggerOccurrence: ...
```

**Implementation:**

1. 首批只实现 schedule 与已有 Feishu event；Webhook、repo 和 CI adapter 等 Owner 出现真实需求后另立任务，不进入首版依赖。
2. Feishu adapter 不建立第二个 WebSocket observer，也不重新私自认证原始消息；它只消费现有主 Bot/Employee canonical ingress 已认证、已绑定 receiving bot principal、tenant、message/event revision、source cursor 且已写 GroupLedger/Journal 的事件。
3. 事件必须有 source id、已认证身份、固定部署 scope 和 freshness。
4. 所有 adapter 只产生 occurrence，不直接创建 session 或调用 Agent。
5. 去重在统一 ingress/projection 完成，不在每个 adapter 私有 map 中完成。
6. canonical event 必须有“聊天正常路由”与“Trigger 订阅”的互斥/并行规则，防止同一消息被普通 Slock 和 Trigger 双 admission。
7. repo/CI 类事件以后通过 authenticated adapter 接入，不在 Agent 内轮询。

**Tests first:**

```python
def test_duplicate_feishu_event_creates_one_occurrence(): ...
def test_adapter_never_dispatches_agent_directly(): ...
def test_stale_event_follows_definition_freshness_policy(): ...
def test_feishu_trigger_consumes_canonical_event_without_second_subscription(): ...
def test_same_message_cannot_double_admit_chat_and_trigger_runs(): ...
def test_non_owner_cannot_create_or_enable_trigger(): ...
```

**Acceptance:**

- 新 Trigger 来源不会新增第二套准入/调度逻辑。
- 未认证 event 不能消耗预算或创建任务。

**Verify:**

```bash
uv run pytest tests/autonomous/integration/test_trigger_adapters.py tests/autonomous/security/test_trigger_authentication.py -q
```

**Commit:** `feat(autonomous): normalize proactive trigger ingress`

### Task C4：建立原子 Admission、去重、Lease 和 Dead Letter

**Files:**

- Create: `src/autonomous/proactive/admission.py`
- Create: `src/autonomous/proactive/leases.py`
- Create: `src/autonomous/proactive/dead_letter.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Test: `tests/autonomous/integration/test_proactive_admission.py`
- Test: `tests/autonomous/chaos/test_proactive_exactly_once_admission.py`

**Required sequence:**

```text
occurrence.observed
  -> authority.checked
  -> atomic Journal transaction frame:
       occurrence.bound(run_id)          [occurrence aggregate]
       run.created(canonical_plan_id)    [run aggregate]
     | rejected | duplicate(existing_run_id) | expired
```

**Implementation:**

1. 使用 `JournalWriter.commit()` 的同一 transaction frame 原子提交 `occurrence.bound` 与 `run.created/plan.compiled`，两条 event 分别携带自己的 aggregate id、expected version/CAS；不能用一个 event 假装维护两个 aggregate，也不能用多个独立 frame 拼成“事务”。
2. lease 有单调 fencing epoch；epoch 必须进入 attempt/effect PREPARED、EXECUTING、adapter invocation、terminal commit 和 outbox frame，每一步都与最新 projection 比较。过期 worker 不但不能提交 Effect，也不能提交迟到结果或消息。
3. duplicate 返回原 run id，不创建第二次执行。
4. 预算不足、owner 撤权、过期、policy 拒绝和暂时故障使用不同状态。
5. 可重试基础设施错误进入有界 retry；不可恢复项进入 dead letter 并通知 owner。

**Tests first:**

```python
def test_duplicate_occurrence_binds_to_one_run_under_concurrency(): ...
def test_stale_lease_holder_cannot_dispatch_effect(): ...
def test_stale_lease_holder_cannot_commit_late_result_or_outbox(): ...
def test_expired_or_revoked_trigger_consumes_no_agent_call(): ...
def test_dead_letter_preserves_reason_and_original_occurrence(): ...
def test_kill_between_each_admission_boundary_never_leaves_consumed_without_run(): ...
def test_atomic_frame_advances_occurrence_and_run_aggregate_versions_together(): ...
```

**Acceptance:**

- “exactly once”只承诺逻辑 admission；远端 Effect 仍使用 prepared/committed/unknown 语义。
- kill/restart 不会重复创建 run。

**Verify:**

```bash
uv run pytest tests/autonomous/integration/test_proactive_admission.py tests/autonomous/chaos/test_proactive_exactly_once_admission.py -q
```

**Commit:** `feat(autonomous): make trigger admission idempotent`

### Task C5：将主动任务接入同一编排与策略门

**Files:**

- Create: `src/autonomous/proactive/planner.py`
- Create: `src/autonomous/proactive/runner.py`
- Modify: `src/orchestration/validation.py`
- Modify: `src/orchestration/ports.py`
- Modify: `src/orchestration/dispatch.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Test: `tests/autonomous/integration/test_proactive_run.py`
- Test: `tests/autonomous/security/test_proactive_authority.py`

**Initial allowed targets:**

- 单 Employee 的单节点 IR。
- Team v2 / Orchestration IR 中已通过 CP-B-Execute 的节点。
- 明确标为 read-only 的分析/汇总任务。

**首版不提供；未来必须有显式 adapter 合同、测试和 Owner 选择：**

- 自动进入用户的 Direct programming session。
- 自动启动 Deep/Spec。
- push、merge、deploy、删除、权限修改等高风险写操作。
- Agent 自己创建新的永久 Trigger。

**Implementation:**

1. GoalTemplate 只填充受控参数，得到完整 Plan。
2. Plan validation 同时验证 authority、backend capability、budget、data egress 和 `PolicyPort`。
3. 每个外部 Effect 在 dispatch 时再次检查最新 authority，防止执行中撤权失效。
4. 首阶段默认 `assist/read-only`；需要写操作时创建 approval node。
5. Agent 可以建议新 Trigger，但只能生成 draft，必须由 owner 明确确认。
6. 复用 Task B5 已通过 production conformance 的 `PolicyPort/EffectDispatchPort`；不得直接重新启用旧内存 `PolicyEngine` 或未二次 anchor 的 legacy `DispatchGate`。
7. Proactive Assist 只允许通过所需 `PermissionProfile` 和 data-egress conformance 的 binding。无法强制 DENY_ALL/READ_ONLY、无法限制额外网络出口的 opaque CLI/ACP 必须运行在可证明的 OS+network sandbox 中，否则不具备 proactive eligibility，即使它可用于人工 Direct Lane。

**Tests first:**

```python
def test_proactive_task_uses_same_plan_and_effect_gate_as_manual_run(): ...
def test_revocation_between_plan_and_dispatch_blocks_effect(): ...
def test_agent_can_only_draft_not_enable_new_trigger(): ...
def test_proactive_trigger_cannot_enter_direct_deep_or_spec_lane(): ...
def test_external_provider_prompt_is_denied_outside_data_egress_scope(): ...
def test_backend_without_permission_and_egress_enforcement_is_ineligible_for_proactive(): ...
```

**Acceptance:**

- 主动入口没有单独的“自动批准”捷径。
- 关闭 proactive runner 后，人工任务和成熟路径正常工作。

**Verify:**

```bash
uv run pytest tests/autonomous/integration/test_proactive_run.py tests/autonomous/security/test_proactive_authority.py -q
```

**Commit:** `feat(autonomous): execute proactive goals through shared gates`

### Task C6：增加 Owner 确认、Budget 和 Kill Switch

**Files:**

- Create: `src/autonomous/proactive/standing_order.py`
- Modify: `src/autonomous/policy/kill_switch.py`
- Create: `src/autonomous/policy/kill_migration.py`
- Modify: `src/autonomous/policy/policy_engine.py`
- Modify: `src/autonomous/gateway/coordinator.py`
- Modify: `src/autonomous/team/coordinator.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Test: `tests/autonomous/security/test_standing_order.py`
- Test: `tests/autonomous/chaos/test_proactive_kill_switch.py`
- Test: `tests/autonomous/unit/test_kill_switch_migration.py`

**Implementation:**

1. 首版写操作使用逐 Run 的 Owner 确认，绑定 action kind、resource scope、risk、次数、预算、有效期和一次性 nonce；approval 不可被另一个 project/trigger/run 重放。
2. `standing_order.py` 是后续扩展：只有 Owner 反复执行同一种低风险动作并明确需要时才启用；其权限不得超过逐 Run 确认，且必须有更短 TTL、次数和预算。首版验收不依赖 standing approval。
3. 不创建 proactive 专属 kill switch。将现有 `src/autonomous/policy/kill_switch.py` 迁移为唯一 Journal-backed `KillPort/KillProjection`，并让 manual orchestration、Employee gateway、Team 和 proactive 共用。
4. scope 分为 system、project、chat、employee、trigger、run；固定 `tenant_key` 只留在内部事件命名空间，不形成单独的发布/运营层级。每次 kill 有单调 epoch，任何较旧 epoch 不能解除或派发。
5. kill 后先阻止新 admission/dispatch，再取消 in-flight；无法确认取消的 Effect 标记 unknown。
6. unkill 需要有权 principal、一次性 nonce 和高风险 scope 的双重确认；restart 时 kill state 在任何 admission 恢复前先 replay。
7. R4 永久拒绝；删除、部署、权限修改等高风险动作首版拒绝，不能用 standing approval 绕过。
8. compat parser 把现有 `kill.switch` JSON 的所有 active scope/epoch 原子迁入 Journal；先验证投影等价，再永久禁用旧 writer。迁移期只有 Journal owner 可写，旧文件只读；代码回滚若看不到新 Journal kill schema 必须 fail closed，不能把系统当作 un-killed。

**Tests first:**

```python
def test_run_approval_cannot_cross_scope_or_replay_nonce(): ...
def test_optional_standing_approval_is_never_broader_than_run_approval(): ...
def test_kill_switch_causes_zero_new_external_calls_after_anchor(): ...
def test_restart_replays_kill_before_trigger_recovery(): ...
def test_r4_action_is_denied_even_with_standing_order(): ...
def test_manual_team_and_proactive_share_one_kill_epoch(): ...
def test_stale_or_replayed_unkill_nonce_cannot_clear_newer_kill(): ...
def test_legacy_active_kill_and_epoch_migrate_without_dual_writer(): ...
def test_rollback_cannot_ignore_newer_journal_kill_state(): ...
```

**Acceptance:**

- Owner 可以一键停止主动能力，而无需关闭主 Bot 和 Direct Lane。
- kill 状态与取消结果可审计。

**Verify:**

```bash
uv run pytest tests/autonomous/security/test_standing_order.py tests/autonomous/chaos/test_proactive_kill_switch.py tests/autonomous/unit/test_kill_switch_migration.py -q
```

**Commit:** `feat(autonomous): enforce owner approvals and kill switches`

### Task C7：设计主动任务卡片、通知节流和例外升级

**Files:**

- Create: `ux/proactive-goal-card.html`
- Create: `ux/proactive-digest-card.html`
- Create: `src/autonomous/proactive/renderer.py`
- Modify: `src/card/session/core.py`
- Create: `tests/autonomous/unit/test_proactive_renderer.py`
- Create: `tests/autonomous/integration/test_proactive_notifications.py`

**UI requirements:**

- Goal/Trigger：owner、范围、下次触发、有效期、预算、模式。
- Run：为何被触发、当前步骤、已用预算、最新产物、风险/审批。
- Controls：pause、resume、run now、edit、revoke、stop。
- Notification policy：立即异常、完成摘要、每日 digest、quiet hours。
- 所有主动消息必须明确标注来源；不能伪装成人工即时请求。

**Implementation:**

1. 先在 `ux/` 产出 Interactive Card 2.0 预览。
2. 卡片分段呈现推理进展摘要，不展示隐藏 chain-of-thought。
3. 合并高频进度，避免卡片洪泛；保留现有执行记录/子任务折叠框合同。
4. 正常运行使用 digest；权限、预算、重复失败和 unknown Effect 立即升级。
5. 移动端和桌面端都验证按钮、折叠和长文本。

**Tests first:**

```python
def test_trigger_card_discloses_scope_expiry_and_budget(): ...
def test_quiet_hours_suppress_progress_but_not_security_exception(): ...
def test_notification_coalescing_preserves_terminal_and_approval_events(): ...
def test_existing_execution_record_fold_is_unchanged(): ...
```

**Acceptance:**

- Agent 主动工作不会把群聊变成日志流。
- 用户可以从一张卡判断“为什么启动、能做什么、花了多少、如何停止”。

**Verify:**

```bash
uv run pytest tests/autonomous/unit/test_proactive_renderer.py tests/autonomous/integration/test_proactive_notifications.py -q
```

**Commit:** `feat(cards): add bounded proactive task controls`

### Task C8：现役 composition、恢复和直接启用

**Files:**

- Create: `src/autonomous/proactive/service.py`
- Modify: `src/autonomous/provisioning/composition.py`
- Modify: `src/feishu/ws_client.py`
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Test: `tests/autonomous/integration/test_proactive_composition.py`
- Test: `tests/autonomous/chaos/test_proactive_restart.py`

**Enablement model:**

```text
PROACTIVE_ENABLED=true|false
each Trigger: disabled | assist_read_only | supervised_write
```

**Implementation:**

1. 实现和依赖检查完成后，当前 Owner profile 默认 `PROACTIVE_ENABLED=true`；服务无 Trigger 时保持空闲。该开关只用于紧急关闭整个主动子系统，不承载发布阶段。
2. 每个 Trigger 创建时默认 `disabled`，只有 Owner 明确启用后才 admission；不存在 `shadow_observe` 或 tenant allowlist。
3. `assist_read_only` 允许满足权限和 data-egress 合同的只读模板。
4. `supervised_write` 仅允许 CP-C-Write 已覆盖的动作，并逐 Run 使用 Owner 确认；模式是授权强度，不是灰度状态。
5. startup 先恢复 Journal/Vault 与 policy/kill/trigger projection，但 trigger 始终保持 fenced；随后严格完成现役 Employee Runtime 的 membership、data/workspace、GroupLedger、Actor mailbox、ingress/router/outbox、unfinished attempts、Team、fire、context、Employee Channel 恢复。只有整体 readiness 与 admission gate 开放后，才启动 adapter due-scan 和 durable scheduler。
6. shutdown 先 fence 新 occurrence/admission，再停止 adapter/scheduler、取消或处置 in-flight，最后按现役 Employee Runtime 反向关闭。
7. B6 已迁移的 `src/autonomous/scheduler/scheduler.py` 是唯一 durable scheduler；仅 manager-only legacy consumer 被隔离，不能另起 proactive scheduler。
8. 故障隔离按层处理：schedule parser、单 adapter 或 proactive worker 故障可降级关闭 proactive；共享 Journal、anchor、Vault、policy projection 或 schema integrity 故障必须让依赖它们的 Employee Runtime 一并 fail closed，不能宣称局部忽略。

**Tests first:**

```python
def test_completed_owner_profile_defaults_service_on(): ...
def test_enabled_service_without_trigger_dispatches_nothing(): ...
def test_new_trigger_is_disabled_until_owner_enables_it(): ...
def test_emergency_disable_starts_no_trigger_worker(): ...
def test_recovery_loads_kill_and_authority_before_due_trigger(): ...
def test_legacy_and_production_scheduler_cannot_both_start(): ...
def test_due_scan_stays_fenced_until_employee_runtime_is_fully_ready(): ...
def test_shared_journal_or_anchor_failure_fails_employee_and_proactive_closed(): ...
```

**Acceptance:**

- 独立 adapter/worker 故障不影响主 Bot、员工 Bot、Direct/Deep/Spec；共享 Journal/anchor/Vault/policy 故障按现役 Employee 安全合同 fail closed，不作过强隔离承诺。
- readiness 明确显示 proactive mode、lag、dead letters 和 disabled reason。

**Verify:**

```bash
uv run pytest tests/autonomous/integration/test_proactive_composition.py tests/autonomous/chaos/test_proactive_restart.py -q
uv run python -m src.main --validate
```

**Commit:** `feat(autonomous): compose owner-enabled proactive runtime`

### Checkpoint CP-C-Assist：Assist/Read-only

- 仅配置的 Owner 可创建/启用 Trigger，且 project/chat scope 显式。
- occurrence 去重、TTL、budget、quiet hours、kill/restart 全通过。
- 除 AuthorityEnvelope 明确允许的 provider 数据外发外，不产生外部可变副作用；只写 GhostAP 自身 Journal/卡片。
- 任何 direct/deep/spec 路由变化都必须先修复；通过后 Automation 入口立即对 Owner 可见。

### Checkpoint CP-C-Write：Owner 确认写入

- 逐 Run Owner approval 的 scope/nonce/expiry 全通过；可选 standing approval 单独验证。
- 写 Effect 都经过 PREPARED/anchor/policy/dispatch/COMMITTED。
- unknown Effect 有人工 reconcile UI。
- 当前 Owner 环境完成撤权、重放攻击、断网和 cancel 演练后，对应可逆写动作直接可用。

---

## 12. Phase D — Evaluation, Skill Evidence and Adaptive Routing

**目标：** 让平台基于证据逐步选对 Agent、组织更好的团队和优化上下文，同时不让统计学习绕过显式选择或安全策略。

**预计投入：** 20–35 engineer-days。

### Task D1：定义不包含隐藏思维链的 Execution Trace

**Files:**

- Create: `src/evaluation/trace.py`
- Create: `src/evaluation/projection.py`
- Create: `src/evaluation/privacy.py`
- Create: `src/evaluation/revocation.py`
- Modify: `src/acp/client.py`
- Modify: `src/orchestration/events.py`
- Test: `tests/evaluation/test_trace.py`
- Test: `tests/evaluation/test_trace_privacy.py`

**Trace fields:**

- route reason、backend/model/capability version。
- queue/start/end/cancel acknowledgment 时间。
- node/attempt/artifact lineage。
- token/usage/cost（仅 provider 有可信数据时）。
- deterministic verifier、review verdict、human correction。
- error code、retry reason、cache provenance。

不保存或展示 provider 隐藏 chain-of-thought；只保存用户可见的进展摘要、结构化 decision 和工具/产物事实。Trace 元数据进入当前部署固定命名空间的 projection；可能含代码、工具参数、纠错文本或用户内容的 payload 只以加密 Blob/Artifact 引用保存，并继承 project/chat ACL、sensitivity、retention 和删除 lineage。当前计划不提供 eval 数据对外导出入口；未来若真实需要导出，再单独设计 Owner 显式动作、secret/source-code scan 和目标约束。

**Tests first:**

```python
def test_trace_contains_binding_timing_and_artifact_provenance(): ...
def test_hidden_reasoning_content_is_not_persisted(): ...
def test_missing_usage_is_unknown_not_zero(): ...
def test_trace_payload_is_encrypted_scoped_and_retention_bound(): ...
def test_trace_has_no_implicit_or_agent_callable_export_path(): ...
```

**Acceptance:**

- 可以区分工具发现失败、模型失败、任务失败和验证失败。
- 不用虚假 `0 token` 污染统计。

**Verify:**

```bash
uv run pytest tests/evaluation/test_trace.py tests/evaluation/test_trace_privacy.py -q
```

**Commit:** `feat(evaluation): add privacy-safe execution traces`

### Task D2：建立离线 Eval 与 Replay Harness

**Files:**

- Create: `src/evaluation/dataset.py`
- Create: `src/evaluation/runner.py`
- Create: `src/evaluation/scoring.py`
- Create: `scripts/run_agent_eval.py`
- Create: `tests/evaluation/test_replay.py`
- Create: `docs/evaluation.md`

**Implementation:**

1. 数据集覆盖 direct coding、Deep、Spec、review、Worktree、Workflow、Team 和 trigger。
2. replay 默认使用 fake provider 和录制 Artifact，不重放外部副作用。
3. 分数分开记录：任务正确性、合同遵守、参数正确、完成率、人工介入、延迟、成本。
4. Agent 友好性与 Backend 能力覆盖使用同一套 capability/conformance 数据，不靠手工表格。
5. 版本化 dataset、prompt、template、backend descriptor 和 scorer。
6. 录制数据只保存在当前 project 的加密本地存储；首版不建设 dataset export，删除/撤销源后本地派生集按 lineage 失效。

**Tests first:**

```python
def test_replay_never_dispatches_recorded_external_effect(): ...
def test_score_distinguishes_contract_failure_from_answer_quality(): ...
def test_dataset_version_and_backend_version_are_part_of_result_key(): ...
def test_revoked_or_tainted_source_is_removed_from_local_eval_dataset(): ...
```

**Acceptance:**

- 每次改 router、prompt、template 或 driver 都能比较前后质量/成本。
- 评估结果可复现，但只作为本地质量比较，不生成发布结论。

**Verify:**

```bash
uv run pytest tests/evaluation/test_replay.py -q
uv run python scripts/run_agent_eval.py --dataset smoke --offline
```

**Commit:** `feat(evaluation): add offline agent replay harness`

### Task D3：把 Slock SkillProfile 升级为证据化 Employee Capability Profile

**Files:**

- Create: `src/evaluation/capability_profile.py`
- Create: `src/evaluation/capability_projection.py`
- Modify: `src/orchestration/events.py`
- Modify: `src/slock_engine/models.py`
- Modify: `src/slock_engine/memory_manager.py`
- Modify: `src/slock_engine/task_router.py`
- Modify: `src/autonomous/team/coordinator.py`
- Test: `tests/evaluation/test_capability_profile.py`

**Required profile fields:**

```python
@dataclass(frozen=True)
class CapabilityEvidence:
    employee_id: str
    tenant_key: str
    project_id: str | None
    task_domain: str
    skill_tag: str
    evidence_ids: tuple[str, ...]
    sample_count: int
    verified_successes: int
    verifier_types: tuple[str, ...]
    confidence: float
    confidence_interval: tuple[float, float]
    observed_from: datetime
    observed_to: datetime
    backend_model_tool_versions: tuple[str, ...]
```

**Implementation:**

1. canonical owner 是由 Journal-backed Run/Attempt/Artifact/verifier 事件可重建的 `CapabilityProjection`；Slock memory/router 和 Team 只能消费这一只读 projection，不能继续写第二份 skill truth。
2. 只有 deterministic verifier、独立 reviewer 或 human outcome 可以提高 verified success；每个聚合值必须列出 evidence ids。
3. Agent 自评和“有输出”只能作为弱信号。
4. project/task-domain、freshness、样本量、backend/model/tool 版本变化必须参与分组或降低可比性；固定 `tenant_key` 只作内部隔离键，禁止跨项目泄漏。
5. profile 是 routing 输入之一，不覆盖显式员工/Backend 选择。
6. 删除/撤销 evidence 后 projection 可重建并移除其贡献；连续失败可以建议降级或人工介入，但不能自动修改 Employee 身份和长期角色文案。

**Tests first:**

```python
def test_self_report_does_not_count_as_verified_success(): ...
def test_stale_or_low_sample_profile_has_low_confidence(): ...
def test_explicit_employee_choice_overrides_profile_ranking(): ...
def test_profile_rebuilds_from_canonical_evidence_and_slock_cannot_write_it(): ...
def test_cross_project_or_revoked_evidence_never_contributes(): ...
```

**Acceptance:**

- “这个 Agent 擅长什么”有证据、样本量和时间范围。
- Slock 与 Team 使用同一个只读 profile projection。

**Verify:**

```bash
uv run pytest tests/evaluation/test_capability_profile.py -q
```

**Commit:** `feat(evaluation): ground employee skills in evidence`

### Task D4：建立有预算、可追溯的 Context Envelope

**Files:**

- Create: `src/autonomous/context/envelope.py`
- Create: `src/autonomous/context/selector.py`
- Modify: `src/autonomous/context/service.py`
- Modify: `src/autonomous/workspace/models.py`
- Modify: `src/autonomous/workspace/projector.py`
- Modify: `src/orchestration/artifacts.py`
- Test: `tests/autonomous/unit/test_context_envelope.py`
- Test: `tests/autonomous/security/test_context_scope.py`

**Required order:**

```text
current thread explicit input
  > current run artifacts
  > project workspace facts
  > scoped group recent context
  > employee L1/L2 memory
```

**Implementation:**

1. 每个 context item 带 source、scope、freshness、sensitivity、token estimate 和 provenance。
2. selector 按 node role/capability 选择上下文，不能默认注入全群历史。
3. context budget exhaustion 产生摘要 Artifact，不悄悄截断关键验收标准。
4. 用户删除、TTL 和 membership 变化立即影响后续选择。
5. coordinator 只接收必要 Artifact 和摘要，不获得所有员工私有记忆。
6. 摘要、Trace、CapabilityEvidence 等派生数据继承 source lineage taint；源被删除/撤权后写 Journal tombstone、删除可删除的加密 Blob，并使派生项不可再选择。Journal 保留最小审计 tombstone，不保留被删除明文。

**Tests first:**

```python
def test_thread_input_outranks_stale_employee_memory(): ...
def test_cross_project_or_expired_context_is_rejected(): ...
def test_context_budget_preserves_goal_and_done_criteria(): ...
def test_coordinator_cannot_read_unselected_private_memory(): ...
def test_source_deletion_taints_and_revokes_derived_summary_trace_and_profile(): ...
```

**Acceptance:**

- 更好的上下文通过选择和 provenance 获得，不靠无限增大 prompt。
- 所有 Agent 可解释“使用了哪些来源”，但不泄露无权内容。

**Verify:**

```bash
uv run pytest tests/autonomous/unit/test_context_envelope.py tests/autonomous/security/test_context_scope.py -q
```

**Commit:** `feat(context): add scoped provenance-aware envelopes`

### Task D5：证据化推荐与 Owner 可选 Auto 路由

**Files:**

- Create: `src/evaluation/router_recommendation.py`
- Modify: `src/agent_session/routing.py`
- Modify: `src/autonomous/team/coordinator.py`
- Test: `tests/evaluation/test_router_recommendation.py`

**Implementation:**

1. recommender 读取 capability、health、profile、cost/latency 和 task requirements。
2. 离线 eval 和诊断模式记录推荐与实际选择的差异，不影响执行；它不形成线上观察期。
3. 相关测试通过后，Auto 直接成为当前 Owner 可选项，仅在用户/Workflow 明确选择 Auto 时使用。
4. 显式 Agent/Employee/model 永远优先。
5. policy、Owner 配置和 capability hard requirements 在 recommendation 之后再次校验。
6. exploration 有比例上限，禁止在高风险任务上随机探索。

**Tests first:**

```python
def test_recommendation_diagnostic_never_changes_actual_backend(): ...
def test_auto_mode_never_selects_backend_missing_hard_capability(): ...
def test_explicit_selection_cannot_be_overridden_by_quality_score(): ...
def test_high_risk_task_has_zero_exploration(): ...
```

**Acceptance:**

- 路由提升必须通过离线 eval、诊断对比和硬能力测试。
- 用户能看到 Auto 选择原因并可固定选择。

**Verify:**

```bash
uv run pytest tests/evaluation/test_router_recommendation.py -q
```

**Commit:** `feat(evaluation): add owner-selectable evidence-based routing`

---

## 13. Phase E — Owner Experience, Reliability and Recovery

**目标：** 让通过工程检查的功能在当前账号直接可用，同时具备可重复的本机体验检查、故障诊断、备份恢复和紧急回退。这里没有灰度、签名发布包或观察窗口。

**预计投入：** 8–15 engineer-days。复用既有收敛计划 Task 22–24 与 Task 26 的本机验证部分；不实施原 Task 25 的租户发布 Gate。

Task E4 是对各功能“检查通过即启用”规则的配置收敛与遗留清理，不允许把启用动作拖到 Phase E；B4、C8 及其他功能任务必须在自己的检查点内完成 Owner 默认可见性。Task E5 只汇总开放后的体验与运维状态。

### Task E1：扩展 Readiness Doctor 和运行指标

**Files:**

- Modify/create from convergence plan: `src/diagnostics/readiness.py`
- Modify: `src/orchestration/diagnostics.py`
- Create: `src/autonomous/proactive/diagnostics.py`
- Modify: `src/feishu/handlers/system.py`
- Test: `tests/test_readiness_agent_platform.py`

**Required checks:**

- Backend catalog/driver/conformance version。
- provider availability 与 model discovery freshness。
- Journal/anchor/Blob/Vault 健康和 replay lag。
- active/queued/unknown/cancel-pending run 数。
- trigger mode、next due、lag、dead letter 和 kill state。
- Employee Channel isolation mode，明确 `bwrap` 与 `process-fallback`。
- queue wait、fsync latency、card update coalescing、provider 429。

**Tests first:**

```python
def test_readiness_never_calls_install_update_or_paid_model(): ...
def test_process_fallback_is_reported_as_no_filesystem_isolation(): ...
def test_unknown_effect_or_dead_letter_degrades_readiness(): ...
def test_enabled_proactive_service_without_triggers_is_healthy_and_idle(): ...
def test_emergency_disabled_subsystem_is_healthy_and_explicit(): ...
```

**Acceptance:**

- readiness 只探测，不修改外部状态。
- 每个 degraded 项给出 owner、影响和恢复动作。

**Verify:**

```bash
uv run pytest tests/test_readiness_agent_platform.py -q
uv run python -m src.main --validate
```

**Commit:** `feat(diagnostics): expose agent platform readiness`

### Task E2：建立单用户工作负载基线

**Files:**

- Create: `scripts/benchmark_agent_platform.py`
- Create: `scripts/run_agent_platform_soak.py`
- Create: `tests/autonomous/acceptance/test_agent_platform_scale.py`
- Create: `tests/orchestration/test_load_profile.py`
- Create: `docs/operations/agent-platform-capacity.md`

**Scenarios:**

1. Owner 日常路径：1 个 Direct 会话、1 个 Employee、1 个 Workflow/Team run、1 个 Trigger，覆盖 cancel/restart。
2. 当前默认上限：最多 8 Employee 的混合 provider、卡片 coalescing、上下文隔离、Journal fsync、文件描述符和内存。
3. 可选容量实验：只有 Owner 真实需要提高上限时才运行 10/50 Employee profile。
4. 可选稳定性诊断：可从 2 小时开始，48 小时仅用于定位偶发 provider 429、飞书限流、网络抖动、凭据刷新或日切问题，不阻塞功能开放。

**Metrics:**

- queue wait p50/p95/p99。
- session start/reuse rate。
- Journal fsync latency/throughput。
- CPU/RAM/process/fd。
- provider error/retry/rate-limit。
- Feishu card update rate/coalescing/drop。
- duplicate occurrence/effect、unknown effect、cancel latency。

**Rules:**

- 默认基线只对当前 Owner、当前主机和当前配置负责，不外推公共容量承诺。
- 每个结果记录硬件、provider 配额、版本和配置，方便回归比较。
- synthetic profile 只用于发现资源问题，不生成发布标签。
- 10/50 profile 或 48 小时 soak 的失败只说明相应规模/时长未经证明；不隐藏已通过日常路径的功能。

**Verify:**

```bash
uv run pytest tests/autonomous/acceptance/test_agent_platform_scale.py tests/orchestration/test_load_profile.py -q -m "not slow"
uv run python scripts/benchmark_agent_platform.py --profile owner-default --smoke
```

**Diagnostic-only（不属于功能完成或开放检查）：**

```bash
uv run pytest tests/autonomous/acceptance/test_agent_platform_scale.py tests/orchestration/test_load_profile.py -q -m slow
uv run python scripts/benchmark_agent_platform.py --profile 8
uv run python scripts/run_agent_platform_soak.py --duration-hours 2 --output-dir /tmp/ghostap-soak
uv run python scripts/benchmark_agent_platform.py --profile 50
uv run python scripts/run_agent_platform_soak.py --duration-hours 48 --output-dir /tmp/ghostap-soak-48h
```

**Commit:** `test(acceptance): add owner workload baseline`

### Task E3：备份、恢复、升级和回滚演练

**Files:**

- Create/complete: `scripts/backup_state.py`（复用 convergence Task 23）
- Create/complete: `scripts/restore_state.py`（复用 convergence Task 23）
- Create: `scripts/verify_agent_platform_backup.py`
- Modify: `src/autonomous/data/projection.py`
- Modify: `src/evaluation/revocation.py`
- Create: `tests/autonomous/acceptance/test_agent_platform_restore.py`
- Create: `docs/operations/agent-platform-recovery.md`

**Backup scope:**

- Journal、anchor、Blob/Data、Vault ciphertext。
- Employee registry/channel binding。
- Orchestration plans/projections。
- Trigger/authority/kill state。
- Artifact schema/catalog version。
- deletion/tombstone high-water mark。

**Implementation:**

1. 备份在停止新 dispatch 或取得一致 snapshot boundary 后进行。
2. restore 到隔离目录先做 chain/hash/blob/schema 验证。
3. 恢复后不自动执行 unknown Effect。
4. catalog/IR/schema migration 必须可重放、版本化和幂等。
5. 回滚代码版本时若不认识新 schema，fail closed，不丢弃事件继续启动。
6. restore 旧备份时，在开放 query/context/eval 前先应用当前 deletion high-water 与 revocation manifest；已删除/tainted Blob 不得从旧备份复活。
7. 明确定义备份保留/到期销毁策略；当前计划没有 eval 数据对外导出路径。

**Tests first:**

```python
def test_restore_rebuilds_same_runs_triggers_and_artifacts(): ...
def test_restore_never_refires_consumed_occurrence(): ...
def test_unknown_schema_blocks_startup_without_mutating_backup(): ...
def test_restore_never_resurrects_deleted_or_tainted_blob(): ...
def test_restore_preserves_deletion_high_water_mark(): ...
```

**Verify:**

```bash
uv run pytest tests/autonomous/acceptance/test_agent_platform_restore.py -q
uv run python scripts/verify_agent_platform_backup.py --source /tmp/ghostap-backup-fixture
```

**Commit:** `test(operations): verify agent platform backup recovery`

### Task E4：直接启用与紧急回退收敛

**Files:**

- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `src/feishu/action_registry.py`
- Modify: `src/feishu/handlers/system.py`
- Create: `tests/test_owner_feature_enablement.py`

**Implementation:**

1. 当前 Owner profile 中，已完成对应工程检查点的能力默认开启并出现在正常菜单；没有 `stable/beta/RC/production` 或 tenant rollout 配置。
2. 每个大型新子系统最多保留一个布尔紧急开关，例如 `WORKFLOW_IR_V2_ENABLED`、`PROACTIVE_ENABLED`；关闭后回到已知兼容路径或只关闭该子系统。
3. 开关不接受百分比、租户列表、时间窗口或 maturity 参数，也不参与路由推荐。
4. 未完成能力仍可显示为 `developing`，动作必须返回具体缺失合同/依赖；不得用“即将开放”掩盖真实错误。
5. 配置迁移删除废弃的 rollout 字段，并对仍存在的旧值给出一次明确迁移提示；正常启动不输出无行动价值的警告。
6. 任一紧急回退都不得影响 Direct/Deep/Spec，也不能跳过 Journal schema、kill 或 unknown Effect 的 fail-closed 规则。

**Tests first:**

```python
def test_completed_capabilities_are_enabled_for_owner_by_default(): ...
def test_no_tenant_percentage_or_release_state_controls_owner_access(): ...
def test_workflow_v2_emergency_off_uses_v1_without_touching_direct_lane(): ...
def test_proactive_emergency_off_leaves_manual_and_protected_lanes_available(): ...
def test_developing_action_reports_concrete_missing_dependency(): ...
```

**Acceptance:**

- 功能完成后不需要再次修改 allowlist、发布标签或环境分组才能体验。
- Owner 能在一次重启内关闭有问题的新子系统，并保留成熟路径。

**Verify:**

```bash
uv run pytest tests/test_owner_feature_enablement.py -q
uv run python -m src.main --validate
```

**Commit:** `feat(product): enable completed owner capabilities directly`

### Task E5：汇总 Owner 端到端体验清单

**Files:**

- Create: `scripts/verify_owner_experience.py`
- Create: `tests/integration/test_owner_experience_contract.py`
- Create: `docs/operations/owner-experience-checklist.md`
- Modify: `docs/goals.md`
- Modify: `README.md`

**Checklist:**

1. **成熟路径：** 分别从飞书真实入口启动 Direct Agent、Deep 和 Spec，确认无额外 planner hop、会话续接/取消和原卡片合同不变。
2. **开发路径：** 已完成的 Worktree、Workflow、Team、Slock 在正常菜单中可见；逐项运行最小真实任务并核对 terminal truth、Artifact、Reviewer、取消与恢复。
3. **Employee：** `/hire`、Bot 身份、会话复用、重启恢复和角色为空时的信息提示正常。
4. **Automation：** Owner 创建、启用、暂停、run now、撤销一个只读 Trigger；写动作必须展示逐 Run 确认，kill 后零新 dispatch。
5. **故障与恢复：** 重启、断网、unknown Effect、备份恢复和紧急回退不复活已删除数据、不重复 Effect。
6. **开放后 UI：** 桌面端和移动端人工查看长文本、按钮和分段进展；现有“并行子任务/执行记录”折叠框不得改变。
7. **诚实记录：** 脚本输出普通 JSON/Markdown 报告，明确 `passed/failed/not_tested`；不签名、不生成 release label，也不把手工未测伪装成失败或通过。

每项能力自己的自动化合同和最小 Owner smoke 在对应检查点完成，随后立即开放；E5 只是把结果汇总到一个持续运维入口。桌面/移动端、长时间和高容量体验在开放后执行，发现问题按缺陷修复或回退对应子系统，不回头建设灰度系统。

**Tests first:**

```python
def test_owner_check_preserves_failed_and_not_tested_items(): ...
def test_direct_deep_spec_checks_are_independent_from_new_subsystems(): ...
def test_completed_capability_is_present_in_owner_menu(): ...
def test_execution_record_and_subtask_folds_are_unchanged(): ...
def test_unknown_effect_or_unacknowledged_cancel_cannot_be_reported_passed(): ...
```

**Acceptance:**

- 一条命令即可运行所有可自动化的 Owner 体验检查，并列出剩余人工项。
- E5 不成为已完成能力的额外总门；检查失败是必须修复的产品缺陷，不触发租户分级或发布晋级流程。
- README、帮助和运行时菜单与实际可用能力一致。

**Verify:**

```bash
uv run pytest tests/integration/test_owner_experience_contract.py -q
uv run python scripts/verify_owner_experience.py --output /tmp/ghostap-owner-experience.json
uv run pytest tests/test_docs_references.py -q
```

**Commit:** `test(acceptance): verify owner experience paths`

---

## 14. 最终产品表面

控制面完成后，默认菜单不应继续按内部模块数量增长，而应围绕以下对象组织：

### 14.1 项目

- 当前目录、代码库、授权、默认 Agent、活动任务。
- 普通编程/Deep/Spec 快捷入口继续存在。
- 项目级任务、预算、Trigger、最近产物和风险事件。

### 14.2 Agent 与 Employee

- **Agent Backend**：Codex/Coco/Claude 等工具、model、状态、真实 capability。
- **Employee**：名字、Bot 身份、Backend 配置、角色、知识空间、能力证据、当前任务。
- 用户可直接进入某个 Backend 编程，也可给长期 Employee 发任务；两者不混为同一对象。

### 14.3 Team

- 成员、协调器、角色、当前 Run、assignment、handoff 和 done criteria。
- Slock 群协作作为 Team 的群聊入口，而不是另一个产品数据库。

### 14.4 Task Center

- Direct/Deep/Spec/WT/WF/Team 的统一只读状态和停止入口。
- active/blocked/waiting approval/unknown/terminal 分组。
- 每项显示实际 owner、backend/model、预算和可恢复性。

### 14.5 Automation

- Trigger/Goal template、下次运行、owner、scope、TTL、budget、mode。
- pause/run now/edit/revoke/kill。
- 主动子系统完成 CP-C-Assist 后立即在 Owner 菜单出现；没有 Trigger 时显示空状态和创建入口，不靠隐藏菜单控制开放。

### 14.6 Approval 与 Audit

- 待批准 Effect、standing authority、unknown Effect。
- route、plan revision、Artifact lineage、成本和最终证据。
- 用户看到的是进展摘要和事实，不展示隐藏思维链。

---

## 15. 优先级、版本切片与投入

### 15.1 推荐顺序

| 优先级 | 范围 | 为什么 |
| --- | --- | --- |
| P0 必须立即做 | Foundation 依赖账本、Task 0.1–0.7 | 先保护成熟路径并修正事实/安全缺口 |
| P1 平台底座 | A1–A6、B1–B3 | 新 Agent 接入、统一可见性和类型化合同，收益高且不要求接管执行 |
| P2 编排可用 | B4–B6、B9a、B10；再并行 B7/B8/B9b | 先直接交付静态耐久 Workflow，再扩展 WT/Team/Slock 和动态协调 |
| P3 主动 Assist | C1–C8 到 CP-C-Assist | 先只读、可停、可审计地主动工作 |
| P4 质量优化 | D1–D5 | 有数据后再做自适应，避免凭感觉路由 |
| P5 本机体验与恢复 | E1–E5 | 完成即开放，同时保证诊断、备份和一键回退 |

### 15.2 可交付版本

**V1 — Connected Agent Platform**

- Phase 0。
- A1–A6。
- B1 统一任务视图。
- 用户收益：直接编程不变；新增 Backend 更快；能力和模型选择更诚实；所有任务更容易查看/停止。

**V2 — Single-user Durable Workflow**

- B2–B6、B9a、B10；B7/B8/B9b 按完成度独立加入。
- CP-B-Execute 通过后 Workflow IR v2 直接出现在当前 Owner 入口，v1 保持紧急回退。
- 用户收益：复杂任务有真实依赖、Reviewer、产物、预算、取消和重启恢复。

**V3 — Proactive Assist**

- C1–C8 至 CP-C-Assist。
- 用户收益：定时/事件驱动的只读巡检、汇总和建议，异常时找人。

**V4 — Owner-supervised Agent Department**

- CP-C-Write、D、E。
- 用户收益：经逐 Run Owner 确认执行有限可逆动作，并能基于证据选择合适 Agent/Employee；每项完成后直接可用。

### 15.3 投入估算

| 阶段 | 增量 engineer-days | 与既有计划重叠 |
| --- | ---: | --- |
| Phase 0 | 15–25 | 与现有正确性/收敛任务部分重叠 |
| Phase A | 25–40 | 与既有 Task 14–16 高度重叠 |
| Phase B | 45–70 | 与既有 Task 13、17–20 部分重叠 |
| Phase C | 30–45 | 主要为新增；不含首版 Webhook/多租户开放状态 |
| Phase D | 20–35 | 主要为新增 |
| Phase E | 8–15 | 与既有 Task 22–24、26 的本机部分重叠 |
| **总增量（未去重）** | **143–230** | 实施前必须用依赖账本去重 |

既有收敛计划估算为 76–123 engineer-days；两份计划不能直接相加，因为 Backend、Task Control、Workflow recovery、doctor 和 backup 有明显重叠，且原租户发布任务已删除。建议先执行 3–5 天的任务去重和基线确认，再给出实际排期。

若由 3–4 个 Agent/工程师并行，按工程依赖串行通过检查点、按独立模块并行实现；每个已完成切片立即交给 Owner 体验。provider 兼容、Journal/Effect 正确性和故障演练是关键路径，增加并行度不能绕过它们。

---

## 16. 成功指标

### 16.1 Direct Lane

- 显式 Agent 请求：额外 planner/router LLM 调用数 = 0。
- catalog 引入前后目标 Backend prompt 次数一致。
- session key、续接、取消、卡片行为合同 100% 通过。
- 启动失败后错误持久化选择 = 0。

### 16.2 Agent 接入

- 新内建 Backend 的身份声明只新增一个 family/binding 声明。
- capability 声明 conformance 覆盖率 = 100%。
- Handler/Engine 新增 backend-name 分支 = 0。
- 发现菜单触发安装/更新次数 = 0。
- 显式选择被静默替换 provider 次数 = 0。

### 16.3 编排

- run/node/attempt/artifact 可 replay 重建率 = 100%。
- required Reviewer 的真实 attempt 覆盖率 = 100%。
- unresolved Effect 下错误终态数 = 0。
- cancel 结果被分为 acknowledged/unknown，禁止虚假成功。
- cache provenance 缺失复用数 = 0。

### 16.4 主动工作

- duplicate occurrence 创建多个 logical run 数 = 0。
- 过期/撤权后新外部调用数 = 0。
- kill anchor 后新 dispatch 数 = 0。
- 无 owner/TTL/budget/authority 的 enabled trigger 数 = 0。
- quiet hours 内非紧急通知泄漏数 = 0。

### 16.5 质量与产品

- 复杂任务完成率和人工接管率按 strategy/backend 分开统计。
- 路由推荐必须报告置信度、样本量和 freshness。
- 用户显式选择覆盖推荐的成功率 = 100%。
- 已完成能力出现在 Owner 正常菜单的比例 = 100%。
- rollout-only 租户 allowlist、百分比和发布状态控制项 = 0。
- “未测试”不计入“通过”。

### 16.6 运维

- Journal/Blob/Vault restore 验证率 = 100%。
- 默认工作负载与按需 soak 中重复 logical admission、session dispatch 和 mediated Effect = 0；opaque nested side effect 以 unknown/reconcile 计，不伪造可见性。
- readiness 读操作产生外部 mutation = 0。
- Owner 端到端清单保留 `passed/failed/not_tested`，不得把 unknown 或未确认取消写成通过。
- 任一新子系统紧急关闭后，Direct/Deep/Spec 合同仍 100% 通过。

---

## 17. 风险、弊端和缓解措施

| 风险 | 为什么会发生 | 影响 | 缓解 |
| --- | --- | --- | --- |
| 统一层变成 God Engine | 试图抽象所有引擎内部步骤 | 成熟路径回归、开发停滞 | 只共享窄 Run/Control/Artifact/Effect 合同；内部算法独立 |
| Direct Lane 变慢 | catalog 查询顺便做 discovery/health/LLM routing | 最高频体验变差 | 本地 O(1) catalog；显式选择优先；纵向调用 recorder |
| ACP/CLI 被假装等价 | capability 过于粗糙 | UI 承诺失真、取消/恢复失败 | capability conformance；CLI 明确 stateless/text-only 降级 |
| IR 限制 Workflow 表达力 | 类型化节点无法覆盖任意 JS | 用户已有 workflow 失效 | v1 兼容、离线 comparator、unsupported 明示、逐模板迁移 |
| 双 Journal/双队列 | 新旧 runtime 并行期过长 | 重复派发、终态冲突 | Journal 单写、owner adapter、迁移后关闭旧消费者 |
| Trigger 风暴 | 重启补跑、重复 event、时钟异常 | 成本和消息洪泛 | deterministic dedupe、bounded misfire、quota、dead letter、quiet hours |
| Coordinator 越权 | JSON prompt 被当成安全边界 | 项目/Shell 被隐式操作 | deny-all tool filter；Artifact 注入；Effect policy |
| 指标驱动错误路由 | 小样本/自评/过期数据 | Agent 被错误分配 | evidence confidence、freshness、离线诊断、显式选择优先 |
| 上下文泄漏 | 为提高质量注入全群/全记忆 | 跨项目或个人数据暴露 | Context Envelope、scope/TTL/provenance、最小选择 |
| 外部副作用无法回滚 | provider/Shell/Feishu 已执行 | restart 后重复或未知 | prepared/committed/unknown、idempotency、人工 reconcile |
| 多副本过早建设 | 对未来规模过度设计 | 复杂度与运维成本激增 | 先用容量证据触发 Hardened ADR |
| 产品入口继续膨胀 | 每个内核增加新命令 | 用户无法形成心智模型 | 围绕项目/员工/团队/任务/自动化/审批组织 |
| 把单用户假设写死进数据模型 | 为省发布逻辑而删除所有 scope | 群消息或未来迁移串线 | 保留固定内部 namespace、Owner/项目/聊天 ACL；只删除发布维度 |

---

## 18. Stop-Expansion Rules

在以下条件满足前，停止新增相邻能力：

1. 一个新 Backend 尚未通过 conformance 时，不新增第二个同类 Backend。
2. Workflow/Team 尚不能真实取消和恢复时，不新增更多动态原语。
3. Slock/Team 仍有双事实源时，不新增自动分配策略。
4. Trigger 尚无 dedupe/TTL/budget/kill 时，不新增新的事件来源。
5. Assist/read-only 的权限、data-egress、Effect、kill 和恢复测试未通过时，不开放 Owner 确认写入。
6. 单机容量尚未测量时，不开始多副本实现。
7. Direct/Deep/Spec 保护合同出现回归时，立即停止平台迁移并回滚。
8. 任何 UI 声称“已评审、已停止、已恢复、已隔离、已完成”，必须有相应运行时证据。
9. 不新增展示隐藏 chain-of-thought 的功能；只展示结构化计划、进展摘要、工具和证据。
10. 不允许聊天用户安装任意 in-process 插件或不在受信 command registry 中的可执行命令；这是安全边界，不是功能开放名单。

---

## 19. 审计发现到计划任务的映射

下表是 2026-07-31 代码快照的计划输入，不是要求无条件重复修复的永久清单。Workflow 等模块当前仍有并行开发；每个任务开始前必须先在目标 branch 上复现、读取相邻未提交改动并检查现有测试。如果后续提交已经关闭某项，应把证据记入依赖账本并跳过对应实现，不得覆盖已有工作。

| ID | 当前发现 | 证据入口 | 负责任务 |
| --- | --- | --- | --- |
| O-01 | Agent 工具列表和分流散落 | `agent_session/factory.py`、`acp/session_factory.py`、各 Handler/Engine | A1–A3 |
| O-02 | Claude CLI model 选择可能未进入真实 argv | `agent_session/claude_cli.py`、factory/startup | 0.3 |
| O-03 | 工具 discovery 可能触发安装/更新 | `acp/helper.py`、provider resolve/update | A4 |
| O-04 | TUI2ACP custom argv/默认 unsafe 边界不清 | `acp/sync_adapter.py`、SystemHandler | A7 |
| O-05 | Workflow UI model map 与 Engine project 分裂 | `workflow_engine/models.py`、`engine.py`、Handler | 0.4 |
| O-06 | Reviewer 选择不保证独立评审 | `workflow_engine/script_gen.py`、Handler | 0.4、B9 |
| O-07 | Workflow Journal 实际更像 invocation cache | `workflow_engine/journal.py` | B5 |
| O-08 | Workflow 总超时会延长，取消无完整确认 | `workflow_engine/bridge.py`、`engine.py` | B6 |
| O-09 | Workflow snapshot 未完整保留活动状态 | `workflow_engine/state_manager.py`、renderer | B10 |
| O-10 | Workflow Handler 与 mixin 同名实现漂移 | `feishu/handlers/workflow.py`、`handlers/mixins/` | 0.4 后的局部清理；禁止与并行修复冲突 |
| O-11 | Worktree timeout/终态/review 偏乐观 | dispatcher、manager、review adapter | 0.5、B8 |
| O-12 | Spec adaptive completion 存在 fail-open | `spec_engine/adaptive_review.py` | 0.6 |
| O-13 | Deep resume 不等于 crash-safe recovery | `deep_engine/engine.py`、`engine_base.py` | B1 诚实投影；不重写 |
| O-14 | Team/Slock 辅助 Agent 工具权限未收口 | Team coordinator、Slock NLI/summary | 0.7 |
| O-15 | Slock 内存 queue 与 Employee Team 并存 | `slock_engine/task_queue.py`、Team projection | B7 |
| O-16 | Trigger/Scheduler 原型未接 production composition | `autonomous/scheduler/`、`domain/goals.py` | C1–C8 |
| O-17 | Channel process fallback 不等于 filesystem isolation | employee channel supervisor | E1、既有 sandbox 安全任务 |
| O-18 | SkillProfile 主要由任务结果和自反馈更新 | Slock memory/router | D3 |
| O-19 | README/欢迎文案与真实生产入口存在漂移 | README、welcome card、SystemHandler | Foundation Task 8、0.1、E4 |
| O-20 | `ws_client`/programming/system/mode/config 仍有生产 Backend 硬编码，闲置 hub 不是 SSOT | Feishu production composition 与 Handler | A2、A3 |
| O-21 | capability 粒度混淆选模/恢复/过滤/取消的请求与真实强制能力 | Session/ACP/CLI adapters | A1、A5 |
| O-22 | 旧 DispatchGate/Policy 未接 Employee production，且执行帧锚定/standing state 不足 | broker gate、policy engine、gateway coordinator | B5、C5 |
| O-23 | 新 Run/Artifact 控制面若无 tenant/ACL 会扩大跨租户读写面 | Tasking 与 Artifact proposal | B1、B3 |
| O-24 | 现有 task/autonomous/Feishu scheduler/control owner 重叠 | `tasking/scheduler.py`、Autonomous scheduler、Feishu control plane | B6、B7 |
| O-25 | opaque ACP/CLI session 内部工具副作用无法由外层 Effect gate 逐项锚定 | Backend session 与 tool callback | A1、B2、B5 |
| O-26 | 主动“只读”仍可能向外部 provider 披露代码/群上下文 | Authority 与 Context/Egress boundary | C1、C5、D4 |
| O-27 | Trace/Eval/摘要派生数据需要 ACL、retention、删除 lineage 和导出许可 | Evaluation/Context proposal | D1–D4 |
| O-28 | 复杂发布矩阵不符合当前单用户场景 | 原 Release/evidence model | E4、E5：Owner 清单与直接启用 |

---

## 20. 每个实施任务的统一工作方式

每个任务执行时都遵循以下顺序：

1. 回读本计划、既有收敛计划的对应任务和最新 `.Memory/Abstract.md`。
2. `rg` 检查已有模式和所有调用方。
3. 先写失败测试；高风险路径增加 contract/chaos/security 测试。
4. 做最小实现，不顺带迁移下一阶段。
5. 运行最相关测试，再按共享影响扩大。
6. 对卡片改动先更新 `ux/` 预览，并保留现有折叠执行记录合同。
7. 更新 `.Memory/{date}.md` 和 `.Memory/Abstract.md`。
8. 按 `docs/commit-message-guidelines.md` 提交。
9. 每个工程检查点单独审查，不把多个检查点混在一个 commit。

通用验证：

```bash
uv run python -m pytest tests/ -q -m "not slow"
uv run python -m pytest tests/ -q
uv run ruff check src/
uv run python scripts/test_inventory.py tests/
uv run python -m src.main --validate
```

相关真实外部行为和 slow 测试不能因耗时而伪装为通过；按需容量/长 soak 没有运行时明确标为 `not_tested`，但不阻塞已通过日常路径的功能直接开放。

---

## 21. 最终建议

### 现在应该做

1. 先完成 Foundation 依赖去重与 Phase 0，保护普通编程/Deep/Spec 并修正当前事实。
2. 优先完成 Agent Catalog + Conformance，而不是继续增加 Agent 名称。
3. 用 Workflow/Team/Slock/Worktree 作为 IR、Journal、Budget、Cancellation 的开发路径；每个闭环完成后立即开放。
4. 在 Task/Effect 事实可靠之前，不上线持续 Trigger。

### 接下来最值得投入

1. Durable Orchestration：这是把“多个 Agent 能并行调用”升级为“可交付复杂任务”的关键。
2. Artifact/Done Criteria：这是减少 Agent 间信息丢失和虚假完成的关键。
3. Proactive Assist：从只读巡检、总结、提醒开始，建立用户信任和本机运维证据。
4. Evidence-based Capability：数据足够后再做 Auto route 和团队自动组建。

### 暂时不要做

1. 不把 Direct/Deep/Spec 重写到统一 IR。
2. 不同时保留 Workflow Journal、Team projection、Slock queue 三套执行事实源。
3. 不让 Agent 在没有 owner、TTL、预算和 authority 时创建永久目标。
4. 不开放任意第三方插件在主进程执行。
5. 不在单机证据不足时投入多副本。

### 产品上限判断

在当前单机、文件 Journal、受信工程主机边界内，GhostAP 有机会成为一个非常强的“可恢复多员工软件交付部门”：既保留与 Codex 等 Agent 直接编程的高效体验，也能将复杂工作交给有身份、有上下文、有审计的 Agent 团队，并让这些团队在有限授权下主动巡检和推进。

它在这一 profile 下不能诚实成为“无人监管、跨系统任意行动、跨区域高可用的自主组织”。达到那一层需要外部一致性、强隔离、KMS/身份、分布式租约、成本治理和相应故障证据。把这条边界说清楚，反而能让 GhostAP 在最有价值、最可控的范围内更快形成可靠产品。
