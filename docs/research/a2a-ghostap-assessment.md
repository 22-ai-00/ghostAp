# A2A 协议与 GhostAP 多 Agent 架构评估

> 调研日期：2026-08-14
>
> 调研范围：A2A 官方协议、官方 SDK/样例，以及 GhostAP 当前 ACP、Workflow、Employee/Team/Gateway 实现。
>
> 来源原则：协议事实只引用 `a2aproject` 官方仓库、规范、SDK 和样例；对 GhostAP 的判断基于当前仓库源码。
>
> 版本基线：A2A 协议 release `v1.0.1`，线上 `protocolVersion` 为 `1.0`；Python SDK release `v1.1.2`。三者是不同版本维度。

## 1. 结论先行

用户提出的方向**基本正确，但抽象需要补全**：

- 一个由 `LLM1 + Codex` 驱动的 Agent，可以作为 A2A Client，把任务交给一个由 `LLM2 + TraeX` 驱动的 A2A Server；第二个 Agent 也可以回传状态、消息和产物。A2A 官方模型明确允许“一个 Agent 充当另一个 Agent 的客户端”，并把服务端 Agent 当作不暴露内部记忆、工具和实现的独立 agentic application。[A2A README](https://github.com/a2aproject/A2A/blob/v1.0.1/README.md)、[Key Concepts](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/key-concepts.md)
- 但 `Agent = LLM + Codex/TraeX` 只描述了**运行时绑定**，不是完整的 A2A Agent。更准确的定义是：

  ```text
  A2A Agent
    = 稳定身份
    + 对外能力/技能声明
    + 认证、授权与策略边界
    + 上下文/记忆与任务生命周期服务
    + 一个内部执行器（例如 LLM + Codex ACP，或 LLM2 + TraeX ACP）
  ```

- A2A **不负责 spawn/provision**。它定义的是已存在、可寻址 Agent 之间怎样发现、鉴权、发消息、跟踪任务、流式收取产物和取消任务。规范的操作集中没有“创建 Agent”“派生子 Agent”“录用员工”“继承预算/权限”“扩展团队池”等操作。因此，“Agent1 自己派生 Agent2”必须拆成两步：先由 GhostAP 受控地创建并准入 Agent2，再通过 A2A 与 Agent2 通信。这是根据 A2A 的规范操作全集作出的架构判断。[A2A Protocol Specification](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)、[Normative protobuf](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)
- 对 GhostAP 最合适的是**混合分层**，不是把 ACP 全部替换为 A2A：

  - ACP 继续负责 GhostAP/Employee 到 Codex、TraeX、Claude 等编码工具后端的会话和工具事件；
  - Journal、Trust/Gateway、Team Coordinator、Workflow 确定性运行时继续负责权限、预算、调度、恢复、验证和最终完成判定；
  - A2A 只放在“独立可寻址 Agent”之间，尤其是跨进程、跨主机、跨框架或第三方 Agent 边界。

- GhostAP 当前最自然的 A2A 服务端粒度是**持久 Employee**，最自然的多 Agent 用法是**受 Coordinator 管控的 Team**。普通 Workflow 的本地临时 `agent()` 调用不值得立刻改成 A2A；将来只需允许冻结 Agent Pool 中的某个成员由 `RemoteA2ADispatchAdapter` 实现。
- A2A 能改善的是互操作性、异构接入、长任务恢复和边界清晰度，不一定改善本地执行速度。对同机短调用增加 HTTP/gRPC、序列化、流状态和重复任务映射，通常会比直接 ACP/进程内调用更慢。
- 这与 ACP 是否“最新”不是同一个问题。当前项目固定 `agent-client-protocol==0.12.0`；A2A 是另一层、另一套协议和 SDK。即使 ACP 升级，也不会自动获得 Agent-to-Agent 的发现、任务和远程互操作语义；引入 A2A 也不要求先替换 ACP。[pyproject.toml](../../pyproject.toml)

建议决策：**值得做一个窄范围 A2A 试点，但不应全量重构。第一阶段只增加 outbound remote Employee adapter；不要动 ACP 核心，不要新增自由 peer mesh，不要先做 push notification。**

## 2. A2A 到底是什么

### 2.1 协议定位

A2A 的目标是让不同框架、不同厂商、运行在不同服务上的不透明 Agent 以统一协议协作。客户端只依赖对方公开的 Agent Card、消息/任务接口和安全要求，不需要知道对方使用什么模型、记忆系统或工具。[A2A README](https://github.com/a2aproject/A2A/blob/v1.0.1/README.md)

官方把参与者区分为：

| 角色 | 协议含义 | 对 GhostAP 的对应 |
| --- | --- | --- |
| User | 发起目标的最终用户或系统 | 飞书用户、API 调用方、上游 Agent |
| A2A Client | 代表用户向远端 Agent 发请求的客户端；自身也可以是 Agent | Team Coordinator、Employee 的受控 outbound adapter |
| A2A Server / Remote Agent | 实现 A2A 端点的独立 agentic application | 持久 Employee facade、第三方 Agent 服务 |

A2A 消息中的 `USER` 与 `AGENT` 角色只表达**客户端到服务端/服务端到客户端的方向**，不是 GhostAP 的员工角色、组织身份或权限主体。身份仍应由传输层认证和 GhostAP 自己的主体映射确定。[Normative protobuf: Role and Message](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)

### 2.2 三层规范和三种标准绑定

A2A 1.0 把协议分成三层：

1. 规范数据模型；
2. 与传输无关的抽象操作；
3. 具体协议绑定。

规范数据模型的权威源是官方 `a2a.proto`。1.0 的标准绑定包括 JSON-RPC、gRPC 和 HTTP+JSON/REST；自定义绑定可以扩展，但必须保持核心语义。Agent Card 的 `supportedInterfaces` 按偏好顺序声明 URL、`protocolBinding`、`protocolVersion` 和可选 `tenant`，客户端选择双方都支持的第一个接口。[A2A Specification](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)、[Custom Protocol Bindings](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/custom-protocol-bindings.md)

这意味着 GhostAP 不应把 A2A 等同于“某个 JSON-RPC 库”。需要在领域层依赖 A2A 语义，在适配器层选择一个绑定。首个 Python 试点可选 JSON-RPC 或 HTTP+JSON + SSE；跨语言且已有 protobuf 基础设施时再评估 gRPC。

## 3. 核心数据模型

| 对象 | 规范语义 | GhostAP 建议映射 |
| --- | --- | --- |
| Agent Card | Agent 的名称、描述、接口、版本、能力、技能、输入输出媒体类型和安全要求 | Employee 的公开、最小化 capability view；不能直接序列化整个内部定义 |
| Message | 一次客户端/服务端通信；由创建方生成 `messageId`，可关联 `contextId`、`taskId`，含多个 Part | assignment 输入、澄清、状态说明；额外保留本地 message/dedupe 映射 |
| Task | 服务端创建的有状态工作单元；含 `id`、`contextId`、状态、产物、历史和 metadata | 一个远端执行尝试，而不是整个 TeamRun；保存 `assignment_id + attempt_id ↔ taskId` |
| Part | 文本、原始字节、URL 或结构化 JSON；可带 MIME、文件名和 metadata | prompt、结构化任务契约、附件引用；全部视为不可信输入 |
| Artifact | Task 的输出；一个或多个 Part，可流式追加 | 代码补丁、评审报告、证据、测试输出；先落地/校验，再进入本地终态判定 |
| Task status event | Task 状态变更 | Journal 中的远端观察事件，不直接成为本地权威状态 |
| Task artifact event | 产物新增/更新，支持 `append`、`lastChunk` | Journal 中的产物分片事件，按 `(taskId, artifactId)` 重组和去重 |

对象和字段的规范定义见 [A2A Specification §4](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md) 与 [a2a.proto](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)。

### 3.1 Task 生命周期

Task 状态包括：

```text
UNSPECIFIED
SUBMITTED -> WORKING
             ├─> COMPLETED   (terminal)
             ├─> FAILED      (terminal)
             ├─> CANCELED    (terminal)
             ├─> REJECTED    (terminal)
             ├─> INPUT_REQUIRED (interrupted)
             └─> AUTH_REQUIRED  (interrupted)
```

终态 Task 不能重新启动。若需要基于旧结果继续或修订，应创建同一 `contextId` 下的新 Task，并可通过 `referenceTaskIds` 引用旧 Task。一个 context 中可以存在多个并行 Task。[Life of a Task](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/life-of-a-task.md)、[a2a.proto TaskState](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)

对 GhostAP 最重要的两个后果：

1. A2A `taskId` 是**服务端生成**的，不能用 GhostAP assignment ID 强行代替；本地必须持久化映射。
2. 远端 `COMPLETED` 只表示远端声称任务完成，不能直接让 GhostAP TeamRun/Effect 完成。GhostAP 仍要校验证据、合并产物、满足 effect finalization，再写本地终态。

### 3.2 Message 与 Task 不是同一层

服务端可以对简单、无状态请求直接返回 `Message`，也可以创建并返回 `Task`。`SendMessageConfiguration.returnImmediately=false` 时，调用默认等待到 Task 到达终态或 `INPUT_REQUIRED`/`AUTH_REQUIRED`；设为 `true` 时可以在任务创建后立即返回。流式调用会先给出 Task/Message，随后发送状态和产物事件。[A2A Specification: Send Message](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)、[a2a.proto SendMessageConfiguration](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)

因此，A2A 不是简单的“Agent 聊天消息总线”。它同时提供短消息和可恢复长任务，但并不提供 GhostAP 的 DAG、团队轮次、工作区锁或验证策略。

## 4. 操作面：A2A 能做什么、不能做什么

A2A 1.0 的核心操作包括：[A2A Specification §3](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)

- Send Message；
- Send Streaming Message；
- Get Task；
- List Tasks；
- Cancel Task；
- Subscribe to Task；
- Create/Get/List/Delete Task Push Notification Config；
- Get Extended Agent Card。

这组操作足以实现 Agent 发现后发任务、持续收进度、断线重订阅、查询/取消长任务，以及离线 webhook 通知。

它没有定义：

- 创建或销毁 Agent；
- 选择模型/编码工具；
- 允许 Agent 任意派生新的 Agent；
- 团队成员录用、池冻结、预算/权限继承；
- 多 Agent DAG、fanout、race、verify、tournament；
- 仓库锁、工作区隔离、命令授权；
- 产物真实性验证或“整个业务已完成”的权威判定。

所以“subagent 自己 spawn Agent2，然后两者用 A2A”中，只有**最后的运行期通信**属于 A2A。spawn/provision、准入和授权必须由 GhostAP 的 Employee provisioning、Trust/Gateway 和调度内核完成。

## 5. Agent Card、发现和能力协商

### 5.1 Agent Card 是服务契约，不是模型配置

Agent Card 至少描述 Agent 名称/说明、一个或多个接口、Agent 自身版本、能力、安全方案/要求、默认输入输出媒体类型和技能。技能包含稳定 ID、名称、说明、标签、示例，以及可覆盖默认值的输入输出媒体类型和安全要求。Card 还可声明 streaming、push notification、extended card、扩展和可选 JWS 签名。[Agent Card protobuf](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)、[Agent Discovery](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/agent-discovery.md)

对于 GhostAP：

- `tool=codex, model=gpt-x` 是 Employee 的内部 executor binding，可以作为受控 metadata 或运维信息，但不应成为 A2A 身份本身；
- 对外 Card 应稳定描述这个 Employee 能做什么，例如 `repository-review`、`targeted-implementation`、`test-diagnosis`；
- 不应在公开 Card 泄露系统 prompt、记忆、token、内部文件路径、完整 permission set、私有模型路由或 ACP 会话信息；
- 如果认证后确有必要提供更详细能力，可使用 Extended Agent Card，而不是把所有信息放进公开 Card。

### 5.2 三种发现方式

官方描述了三种发现策略：[Agent Discovery](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/agent-discovery.md)

1. `/.well-known/agent-card.json` 的标准 well-known 位置；
2. 组织内注册表/目录；
3. 直接配置已知 Card URL/内容。

A2A 核心并不标准化“全局 Agent Registry API”。GhostAP 已有 tenant-aware、Journal 投影的 Employee Registry，因此第一阶段应把它保留为准入和身份权威，只为允许的远端成员保存已验证 Card、endpoint、信任锚与绑定快照。不要因为 A2A 有发现机制就让运行中 Agent 从公网自由搜索并加入任何新成员。

### 5.3 多租户字段不是授权

同一服务可按 URL 子路径、认证身份或 AgentInterface 的可选 `tenant` 做路由。若 Card 选中的接口带 `tenant`，客户端必须原样回传；这个字段是服务端解释的**不透明路由值**，协议不赋予它授权语义。[Multi-tenancy](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/multi-tenancy.md)、[a2a.proto AgentInterface](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)

GhostAP 必须继续使用认证主体、owner/tenant 映射和 action matrix 作授权，不能把“tenant 字符串匹配”当成访问控制。

## 6. 同步、流式和异步 push

### 6.1 SSE / server streaming

标准流式操作允许服务端持续发送 Task、状态和 Artifact 更新；已有非终态 Task 可通过 Subscribe to Task 恢复订阅。订阅开始时先返回当前 Task 快照，有助于避免先 Get 再 Subscribe 之间的竞态。HTTP/JSON-RPC 使用 SSE，gRPC 使用 server streaming。[Streaming and Asynchronous Operations](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/streaming-and-async.md)、[A2A Specification](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)

这对 GhostAP 的远程 Employee 很有价值：WebSocket/飞书连接断开不需要终止远端工作，Coordinator 可凭 Journal 中的 task mapping 重订阅或 `GetTask` 对账。

### 6.2 Push notification

Push 模式允许客户端登记 HTTPS webhook，服务端在 Task 更新时 POST 通知。配置可带客户端校验 token 和服务端访问 webhook 所需认证信息；服务端必须验证目标 URL、防止 SSRF/DDoS，并向 webhook 认证，接收端也要认证服务端、防重放。完整权威快照通常仍通过 `GetTask` 获取。[Streaming and Asynchronous Operations](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/streaming-and-async.md)、[A2A Specification: Push Security](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)

GhostAP 首期不应启用 push，原因是它新增：

- 一个可被远端调用的入站 HTTP 面；
- 服务端向用户提供 URL 发请求的 SSRF/出站访问风险；
- webhook token、签名、重放和轮换问题；
- notification 与 `GetTask` 对账的重复/乱序处理。

首期用 SSE + 有界重连 + `GetTask` 对账即可；只在出现真正的超长离线任务需求后再单独做 push 安全设计。

## 7. 扩展机制与“团队协议”

A2A 扩展由 URI 标识并在 Agent Card 中声明。客户端通过 `A2A-Extensions` header/metadata 显式选择；若服务端要求某扩展而客户端不支持，请求应被拒绝。扩展可以定义额外 metadata、交互 profile、方法或状态语义，但不能改变核心数据结构和枚举；破坏性版本应换新 URI。[Extensions](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/extensions.md)

官方组织为扩展和自定义 binding 定义了“official/experimental”治理等级；SDK 是否支持扩展是可选的，默认不应为了通过核心一致性而打开。[Extension and Binding Governance](https://github.com/a2aproject/A2A/blob/main/docs/topics/extension-and-binding-governance.md)

官方 samples 仓库曾提供 Agent Gateway Protocol（AGP）提案，描述 capability/policy routing 和层级 Autonomous Squads，概念上很接近“团队 Agent 通过网关协作”。但它位于 samples 的 `extensions/agp`，文本本身也以 proposal 表述，并非 A2A 核心规范或已稳定发布的正式扩展。因此它可以作为设计参考，不能直接成为 GhostAP 的生产依赖。[AGP sample proposal（固定历史提交）](https://github.com/a2aproject/a2a-samples/blob/a1a80ebabc2229c81d3f38b3b4a0e3909e33e09d/extensions/agp/spec.md)

若 GhostAP 试点后确认核心 metadata 不够，建议定义一个很小的内部协调扩展，例如：

```json
{
  "runId": "...",
  "planVersion": 3,
  "workUnitId": "...",
  "attemptId": "...",
  "purpose": "review",
  "inReplyToWorkUnitId": "...",
  "deadline": "2026-08-14T12:00:00Z",
  "budgetSlice": {"maxTurns": 4},
  "acceptanceSchema": "ghostap://schemas/review-result/v1",
  "authorityRef": "journal-event-or-policy-ref"
}
```

这里仅传最小的关联与约束引用，不传原始凭据、完整内部权限或可任意扩权的数据。扩展 metadata 仍是远端输入，不能绕过本地 admission/gateway。

## 8. 认证、授权与安全边界

### 8.1 A2A 提供互操作钩子，不替代 Trust/Gateway

生产接口应使用 HTTPS/TLS，客户端验证服务端证书。Agent Card 可声明 API key、HTTP auth、OAuth 2.0、OpenID Connect、mTLS 等安全方案和 requirement；凭据通过标准请求 header/metadata 传输。服务端对每次请求进行认证，具体授权仍由实现按 skill、action、资源和租户决定，并应遵循最小权限。[Enterprise Ready](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/enterprise-ready.md)、[A2A Specification: Security](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)

Task 中若需要补充认证，协议使用 `AUTH_REQUIRED` 表达；凭据应通过带外安全渠道取得，而不是把密钥塞进普通 Message/Artifact。跨 Agent 转发用户 credential chain 会扩大委托和泄漏风险，GhostAP 应优先使用每个 Agent 自己的 service identity 与短时、受众受限 token。

A2A 没有定义 GhostAP 的 R4 风险、assist 只读、命令 allow/deny、一次性授权或 Effect 锚定。因此：

```text
A2A authenticated != GhostAP authorized
A2A Task COMPLETED != GhostAP verified/finalized
Agent Card skill advertised != action admitted
tenant matched != tenant authorized
```

### 8.2 所有远端内容都是不可信输入

官方多 Agent 样例特别提醒：远端 Agent Card 的描述、技能和消息可能包含 prompt injection，不应无条件拼接到主 Agent prompt。该样例同时明确是演示，不是生产级实现。[Airbnb multi-agent sample](https://github.com/a2aproject/a2a-samples/tree/6603ba3f2c31a7ef33e70b9d8b5b5f8be42ac9a3/samples/python/agents/airbnb_planner_multiagent)

GhostAP 至少需要：

- Card 来源 allowlist、TLS 校验、可选 JWS/组织信任锚、Card 内容 schema 和大小限制；
- Card 更新的 ETag/版本对账与 binding freeze；运行中不能因远端 Card 变化静默换模型/身份/权限；
- Message/Part/Artifact 的内容、MIME、大小、数量和 schema 校验；
- URL Part 的协议/主机/IP allowlist、DNS rebinding 防护、下载上限、超时与恶意文件扫描；
- 原始字节解码上限与落盘路径隔离；
- prompt boundary，明确把远端描述和产物标记为 data/evidence，而不是上级指令；
- 远端产物在合并、执行或提交前经过本地验证器和 action matrix；
- 日志脱敏，不记录 bearer token、webhook credential 或完整私密 Card。

### 8.3 幂等与重试不能交给协议想当然处理

A2A 规范规定 Get 类操作天然幂等，Cancel 和删除 push config 是幂等操作，但 **Send Message 仅 MAY 幂等**；服务端可以使用 `messageId` 检测重复，但不是必须。Push 通知也可能重复。[A2A Specification: Idempotency](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)

因此 GhostAP 必须继续让 Journal 成为唯一事实源，并为 outbound 调用持久化：

```text
(run_id, assignment_id, attempt_id)
  -> client_message_id
  -> remote_agent_identity/card_digest
  -> remote_context_id
  -> remote_task_id          # 收到服务端响应后补写
  -> last_observed_event/digest
  -> local verification/finalization state
```

断线或超时时，先依据已知 taskId 查询/订阅；没有 taskId 时，使用相同 messageId 做受控重试，但仍要容忍服务端重复执行。对具有外部副作用的任务，应让 GhostAP Effect/idempotency key 在业务层兜底，而不是依赖 A2A Send Message。

## 9. 官方 SDK、样例和成熟度

### 9.1 版本要分三层看

| 层 | 当前调研基线 | 含义 |
| --- | --- | --- |
| 规范仓库 release | `v1.0.1`，2026-05-28 发布 | 规范文档和修订 release |
| wire `protocolVersion` | `1.0` | AgentInterface 在网络上协商的协议版本；patch release 不写成 `1.0.1` |
| Python SDK package | `v1.1.2`，2026-07-22 发布 | SDK 自身版本；README 声明实现 A2A 1.0，并兼容 0.3 |

来源：[A2A v1.0.1 release](https://github.com/a2aproject/A2A/releases/tag/v1.0.1)、[AgentInterface protobuf](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)、[a2a-python v1.1.2 release](https://github.com/a2aproject/a2a-python/releases/tag/v1.1.2)、[a2a-python v1.1.2 README](https://github.com/a2aproject/a2a-python/blob/v1.1.2/README.md)。

不要把 Python 包版本 `1.1.2` 填进 Card 的 `protocolVersion`，也不要把 A2A 版本与 GhostAP 的 ACP Python 包版本混为一谈。

### 9.2 SDK 能力

官方列出 Python、Go、JavaScript、Java、.NET 和 Rust SDK，并提供 samples、Technology Compatibility Kit（TCK）和 Inspector。[Official SDK index](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/sdk/index.md)、[A2A TCK](https://github.com/a2aproject/a2a-tck)、[A2A Inspector](https://github.com/a2aproject/a2a-inspector)

Python SDK `v1.1.2` 支持 JSON-RPC、HTTP+JSON/REST、gRPC 的客户端/服务端实现，提供 Card resolver/client factory、`AgentExecutor`、request handler、TaskStore、流式 EventQueue，以及可选 Starlette/FastAPI/gRPC/数据库/OTel 依赖。它要求 Python 3.10+，与 GhostAP 当前 Python 3.11+ 基线兼容。[a2a-python README](https://github.com/a2aproject/a2a-python/blob/v1.1.2/README.md)、[AgentExecutor interface](https://github.com/a2aproject/a2a-python/blob/v1.1.2/src/a2a/server/agent_execution/agent_executor.py)

但 SDK 只是协议 plumbing，不应成为第二个领域事实源：

- 若 GhostAP 将来提供 inbound A2A Server，SDK TaskStore 应做成 Journal projection/adapter，或只缓存可从 Journal 重建的数据；
- 不应让内存 TaskStore 与 Journal 分别维护互相冲突的终态；
- SDK 的 push sender、auth hook 或默认 handler 不能替代 SSRF policy、action matrix、quota 和审计。

### 9.3 官方样例能证明什么

- Hello World 展示了 Agent Card、`AgentExecutor`、默认 request handler、TaskStore 和应用路由的最小服务端结构。[Hello World sample](https://github.com/a2aproject/a2a-samples/tree/6603ba3f2c31a7ef33e70b9d8b5b5f8be42ac9a3/samples/python/agents/helloworld)
- Airbnb 多 Agent 样例展示了 host agent 发现多个远端 Card、创建客户端、路由调用并综合结果，证明“Agent1 调 Agent2/Agent3”符合 A2A 模型；它也明确不是生产级安全实现。[Airbnb multi-agent sample](https://github.com/a2aproject/a2a-samples/tree/6603ba3f2c31a7ef33e70b9d8b5b5f8be42ac9a3/samples/python/agents/airbnb_planner_multiagent)
- A2A + MCP 样例把 MCP 用于工具/registry，把 A2A 用于运行期 Agent 调用，体现了内外两层协议可以组合。[A2A MCP sample](https://github.com/a2aproject/a2a-samples/tree/6603ba3f2c31a7ef33e70b9d8b5b5f8be42ac9a3/samples/python/agents/a2a_mcp)

官方文档比较的是 MCP 而非 ACP：MCP 主要解决 Agent 到工具/资源，A2A 解决独立、不透明 Agent 之间的协作。[A2A and MCP](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/a2a-and-mcp.md) 将这个分层类比到 GhostAP 是架构推论：**ACP 位于 Agent 内部的编码工具会话边界，A2A 位于 Agent 外部的对等服务边界。**

## 10. GhostAP 当前架构映射

### 10.1 ACP：保留，不替换

当前 [ACP provider](../../src/acp/provider.py) 和 [ACP session](../../src/acp/session.py) 负责：

- 选择并启动 Codex、TraeX 等编码后端进程；
- 建立/恢复/取消后端 session；
- 设置模型和配置；
- 传 prompt，接收工具调用、文件/终端事件和流式输出；
- 处理 ACP 权限和 provider 特定能力。

这些是“宿主 ↔ 编码 Agent runtime”的细粒度会话语义，A2A 的 Message/Task/Artifact 不提供等价的 terminal、file tool、permission request、session config/resume 合约。强行用 A2A 替换 ACP 会丢失能力或重新发明一层私有扩展。

正确关系：

```text
Remote caller
    │ A2A
    ▼
GhostAP Employee (稳定 Agent 身份与任务服务)
    │ ACP
    ▼
Codex / TraeX / Claude / ... (内部执行后端)
```

ACP 内部观察到的 Codex/TraeX subagent 也不适合立刻逐个包装成 A2A Server：这些子 Agent 往往由 provider 临时创建，缺少 GhostAP 权威身份、独立任务服务、稳定 endpoint 和权限边界；现有 [ACP collaboration reconciliation](../../src/acp/collaboration.py) 也把部分内部 subagent 状态视作非权威观察。

### 10.2 Workflow：保留确定性运行时，只增加可选远端成员

当前 Workflow 已有：

- 冻结的 `agent_id -> tool/model/profile/effort` 绑定和重复检查：[agent_pool.py](../../src/workflow_engine/agent_pool.py)；
- 运行中不允许越过已确认池的 run spec：[run_spec.py](../../src/workflow_engine/run_spec.py)；
- 每个 `agent()` 通过 ACP/CLI 短会话执行，含 schema 修复、重试和关闭：[executor.py](../../src/workflow_engine/executor.py)；
- `classify/fanout/verify/generate/tournament/loop/sequence/race` 等确定性控制原语。

A2A 只解决单个远端成员如何被调用，不解决这些控制原语、池冻结或全局 `MAX_TOTAL_AGENTS`。若把每个同机临时调用都改成 A2A，会产生两套 task lifecycle、额外网络开销和更复杂的失败恢复。

仓库的 [Dynamic Multi-Agent Orchestration V3 计划](../2026-08-12-dynamic-multi-agent-orchestration-v3-plan.md) 已提出 backend-neutral `DispatchPort`，并坚持 planner 不能直接 dispatch 或扩池。A2A 最适合成为未来第三种 adapter：

```text
DispatchPort
  ├─ EphemeralACPAdapter
  ├─ EmployeeDispatchAdapter
  └─ RemoteA2ADispatchAdapter   # 新增；只接受冻结池内成员
```

即：Workflow 可以支持 remote pool member，但不应让脚本通过 Card discovery 临时吸收池外 Agent，也不应把 A2A 变成新的编排内核。

### 10.3 Employee：最佳 A2A Agent 边界

当前 [EmployeeDefinition](../../src/autonomous/domain/employees.py) 已包含稳定 `agent_id`、tenant/owner、工具/模型/profile/effort、角色/persona、capabilities、permissions、budget、bot principal 和 groups；[AgentIdentity](../../src/autonomous/workforce/identity.py) 与 [ProjectedAgentRegistry](../../src/autonomous/workforce/registry.py) 提供后端无关、tenant-aware、由 Journal 投影的身份和绑定校验。

[EmployeeSessionHost](../../src/autonomous/runtime/session_host.py) 将 Employee 的工具/模型/工作区绑定到 ACP session，并安装 tool filter/sandbox；[EmployeeActor](../../src/autonomous/runtime/employee_actor.py) 提供持久 mailbox、串行 assignment、warm session、取消和 Journal 事件锚定。

这已经接近一个完整 A2A Agent 所需的内部构成：

| A2A 对外概念 | Employee 内部来源 |
| --- | --- |
| 稳定 Agent 身份 | AgentIdentity / ProjectedAgentRegistry |
| Agent Card skills | Employee capabilities 的公开映射 |
| 安全要求 | tenant/owner/principal + Trust/Gateway policy 的公开认证要求 |
| Task 执行 | EmployeeActor mailbox/assignment |
| 内部 executor | EmployeeSessionHost -> ACP -> tool/model |
| 持久恢复/审计 | Journal |
| 取消 | actor/backend cancel，经 Gateway 授权 |

因此建议：每个需要跨边界暴露的持久 Employee 可以拥有独立 Agent Card/tenant route；Card 描述 Employee 的服务能力，而不是把它简化成某个 model/tool tuple。

### 10.4 Team：最佳 A2A 协作边界

当前 [Team models](../../src/autonomous/team/models.py)、[Team service](../../src/autonomous/team/service.py)、[Team coordinator](../../src/autonomous/team/coordinator.py) 和 [Employee team gateway](../../src/autonomous/gateway/team.py) 已负责 TeamRun、Assignment、轮次/fanout/handoff 上限、冻结绑定校验、提交/结果/取消/通知和持久 Employee actor 调度。

A2A 映射应是：

```text
一个入站 A2A Task（用户让 GhostAP Team 完成目标）
    ↕ 不是一一对应
一个本地 TeamRun
    ├─ TeamAssignment A -> 本地 Employee assignment
    ├─ TeamAssignment B -> 远端 A2A Task B1
    └─ TeamAssignment C -> 远端 A2A Task C1
```

`TeamAssignment/attempt ↔ remote taskId` 必须持久化。一个 TeamRun 可能产生多个远端 Task；一个远端 Task 也只应代表一个明确、可取消和可验证的执行尝试。不要让 A2A Task 的终态反向覆盖整个 TeamRun。

## 11. 用户设想的精确判定

### 11.1 “Agent1 = LLM1 + Codex，Agent2 = LLM2 + TraeX”

**作为 executor binding 是对的，作为 Agent 身份定义不完整。**

可以实现为：

```text
Employee A1
  identity = reviewer-codex-01
  skills = [repository-analysis, implementation]
  policy = repo-RW-with-tests
  memory/context = Journal + project context
  executor = GPT-x + Codex over ACP

Employee A2
  identity = specialist-traex-01
  skills = [frontend-review, targeted-fix]
  policy = repo-branch-scoped
  memory/context = Journal + project context
  executor = LLM2 + TraeX over ACP

A1 --A2A Message/Task/Artifact--> A2
```

A2A 对 A2 的内部 `LLM2 + TraeX` 完全不感知，这正是它的“opaque agent”价值。

### 11.2 “subagent 自己派生 Agent2”

**不能仅靠 A2A 实现，也不建议允许任意自派生。** 推荐流程：

1. A1 提交一个结构化“需要某能力”的 spawn/hire 请求；
2. GhostAP provisioning/trust 层校验发起者是否有权创建员工、允许哪些工具/模型、权限/预算/工作区和存活时间；
3. 创建 Employee A2，写 Journal，注册稳定 identity，启动/绑定 endpoint；
4. Team/Workflow admission 把 A2 放入本次冻结 pool；
5. 之后 A1 或 Coordinator 才能通过受控 A2A adapter 向 A2 发任务；
6. A2 想继续派生 A3 时重复相同准入流程，不能把 A1 的权限凭据直接传下去。

A2A 负责第 5 步，不负责第 1–4 步。

### 11.3 “团队内都用 A2A 会不会更好”

结论是**跨边界成员更好，本地临时成员未必**：

| 场景 | 是否用 A2A | 原因 |
| --- | --- | --- |
| 远端/第三方专业 Agent | 强烈适合 | 标准发现、认证、任务、流式产物和取消；隔离内部实现 |
| 跨主机的持久 Employee | 适合 | 断线恢复、独立升级、跨语言/框架 |
| 向外部系统暴露 GhostAP Employee/Team | 适合，后期做 | A2A Server 是清晰标准接口，但增加入站安全面 |
| 同进程/同主机、秒级 Workflow 临时调用 | 暂不适合 | 直接 ACP/adapter 更轻；A2A 增加延迟和双重状态 |
| GhostAP 到 Codex/TraeX runtime | 不适合替代 ACP | A2A 不含工具事件、权限请求、会话配置等 ACP 语义 |
| provider 内部自动生成的短命 subagent | 默认不适合 | 缺稳定身份、endpoint、任务服务和权威生命周期 |

## 12. 推荐目标架构

```text
                         ┌──────────────────────────────┐
Feishu / API / A2A in ──>│ Admission + Identity + Trust │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │ Team/Workflow Kernel          │
                         │ Journal = sole source of truth│
                         │ pool / budget / DAG / verify  │
                         └──────────────┬───────────────┘
                                        │ DispatchPort
                  ┌─────────────────────┼─────────────────────┐
                  │                     │                     │
        ┌─────────▼─────────┐ ┌────────▼─────────┐ ┌─────────▼──────────┐
        │ Ephemeral ACP     │ │ Employee Actor   │ │ Remote A2A Adapter │
        │ Adapter           │ │ Adapter          │ │ Card + Client      │
        └─────────┬─────────┘ └────────┬─────────┘ └─────────┬──────────┘
                  │ ACP                │ ACP                 │ A2A
        ┌─────────▼─────────┐ ┌────────▼─────────┐ ┌─────────▼──────────┐
        │ Codex/TraeX/...   │ │ Codex/TraeX/...  │ │ Remote Agent Server│
        └───────────────────┘ └──────────────────┘ │ (its internals opaque)
                                                   └────────────────────┘
```

如果以后提供 inbound A2A Server，它应是同一 admission/Journal/domain service 的 facade 或隔离 sidecar，而不是另起一个 TaskStore 和独立完成判定：

```text
A2A Server handler -> authenticated principal -> admission/gateway
                   -> Journal-backed TeamRun/Employee assignment
                   -> A2A Task projection + event stream
```

GhostAP 当前主要是出站 WebSocket 服务，没有现成公开 ASGI 入站面；因此 outbound-only 试点能以最小攻击面验证价值。入站 A2A 最好后置，并考虑独立进程/sidecar、单独端口和网络策略。

## 13. Team 通信拓扑：Coordinator-mediated，不做自由 peer mesh

推荐星型/受控网关：

```text
                  Team Coordinator / Kernel
                  /          |           \
            A2A Task     Local ACP     A2A Task
               /             |             \
        Remote Agent A   Employee B   Remote Agent C
```

Agent 之间看起来可以“互相委派”，但每次 outbound 调用都经过一个强制边界：

- 目标 Agent 必须在当前冻结 pool；
- binding/Card digest 与准入时一致；
- 总预算、fanout、handoff、deadline 和 cancel 由 kernel 统一管理；
- workspace/repository 权限由 Gateway 校验；
- A2A credential 由 adapter 注入，不能暴露给 LLM；
- 所有状态与 Artifact 都先记 Journal，再交给 Planner/Verifier；
- 远端 Agent 不可凭一条 Message 扩大团队、换模型或扩大权限。

这保留 A2A 的异构互操作优势，也保留 GhostAP 已有的确定性控制和 fail-closed 安全。自由全互联 peer mesh 会让预算、循环委派、取消传播、任务归属和审计迅速失控；A2A 本身没有为这些问题提供完整治理。

## 14. 建议的状态与错误映射

| A2A 观察 | GhostAP 本地动作 |
| --- | --- |
| direct Message | 记录为结果候选；按 assignment schema 验证，不自动视为 TeamRun 完成 |
| SUBMITTED / WORKING | 写远端观察事件，刷新 lease/activity；不改变权威 Effect 锚定规则 |
| INPUT_REQUIRED | 若允许自动恢复，Coordinator 生成有界 follow-up；否则按本地策略明确失败，不能无限等用户 |
| AUTH_REQUIRED | 只走受控带外 credential flow；无授权则拒绝/失败，不把 secret 放进 prompt |
| COMPLETED | 获取完整 Task/Artifact，验证、合并、写本地完成；验证失败可新建同 context 的修订 Task |
| FAILED / REJECTED | 记录原因，按本地 retry/fallback 策略决定新 Task；不复活旧终态 Task |
| CANCELED | 与本地取消意图对账；重复 cancel 应容忍 |
| stream 断开 | 先 Subscribe/GetTask；依据 task mapping 对账，不盲目 SendMessage |
| 未取得 taskId 的超时 | 同 messageId 有界重试，同时按“可能已执行”处理副作用风险 |

## 15. 分阶段落地方案

### Phase 0：ADR 与领域契约，不接网络

1. 明确协议基线：A2A spec release `v1.0.1`、wire `1.0`、Python SDK pin `a2a-sdk==1.1.2`；单独管理 ACP 依赖。
2. 定义本地 `RemoteAgentDescriptor`、`RemoteTaskHandle`、`RemoteAgentDispatchPort`，不让领域层导入 SDK 类型。
3. 定义 `assignment/attempt ↔ messageId/contextId/taskId` 的 Journal 事件和投影。
4. 写清远端 `COMPLETED` 只是 claimed completion，Verifier/finalization 才是本地完成。
5. 明确 Card trust、credential ownership、URL/file policy 和数据保留。

### Phase 1：只做 outbound `RemoteA2ADispatchAdapter`

1. 用固定配置或内部 registry 指定一个已信任的远端 Agent Card，不做开放公网发现。
2. 校验 Card、选择 `protocolVersion=1.0` 的 JSON-RPC 或 HTTP+JSON 接口，冻结 Card digest/binding。
3. 实现 Send Streaming Message、GetTask、Subscribe、Cancel；先不实现 push。
4. 把每个流事件先转换成领域事件并写 Journal，再更新 UI/Coordinator。
5. 将 adapter 放到未来 DispatchPort 后，只允许冻结 pool 的 remote member 被调用。

### Phase 2：两 Employee 受控试点

选择一个可验证、低副作用的真实任务：

```text
Codex-backed Employee A
  -> 生成或分析一个变更
  -> 通过 A2A 委派独立 review 给 TraeX-backed Employee B
  <- B 流式返回 review Artifact
  -> A 本地综合；Verifier 检查证据/测试
```

先让 B 是独立进程或测试服务，以真实覆盖网络断开、取消和重订阅；不要只是进程内 loopback benchmark。整个 pool 在 run 开始前冻结，B 无权自行增加 C。

### Phase 3：把远端成员接入 Team/Workflow

- Team：由 Coordinator 创建/跟踪远端 Task，保留现有 assignment/fanout/handoff/budget 上限。
- Workflow：仅把某些 AgentBinding 的 backend 标为 remote A2A；JS 原语和 pool enforcement 不变。
- 增加跨运行恢复：进程重启后从 Journal 重建 remote task handles 并 Get/Subscribe 对账。

### Phase 4：可选 inbound A2A facade

仅在确有外部系统要调用 GhostAP Employee/Team 时实施：

- 独立服务面、TLS/OAuth2 或 mTLS、速率/配额、tenant principal mapping；
- Agent Card 公开最少信息，敏感技能放 Extended Card；
- SDK TaskStore 只能是 Journal-backed adapter/projection；
- A2A Cancel 通过 Gateway 进入现有 actor/backend cancel；
- 用 TCK 检查规范 MUST，并用 Inspector 做 Card/对话/原始消息检查。

### Phase 5：确有缺口后再做扩展和 push

- 先用 core Task metadata + Artifact schema；只有多实现互通需要稳定语义时才定义 URI extension。
- push notification 单独威胁建模，包含 SSRF、DNS/IP policy、回调认证、重放、重复/乱序和 secret rotation。
- AGP 只做参考，等其进入正式治理层级、版本稳定且与 GhostAP 约束契合后再评估依赖。

## 16. 试点验收矩阵

### 16.1 功能与一致性

- 能获取/缓存/更新 Agent Card，正确选择 `protocolVersion=1.0` 的受支持 binding；
- Message、Task、Part、Artifact 和流式 append/lastChunk 正确映射；
- 服务端生成 taskId 后立即持久化 assignment mapping；
- stream 中断后可 Subscribe/GetTask 恢复且不重复完成；
- Cancel、FAILED、REJECTED、INPUT_REQUIRED、AUTH_REQUIRED 均到达明确本地终态或有界恢复；
- 远端 `COMPLETED` 必须经过本地 schema/evidence/test 验证；
- GhostAP 重启后能从 Journal 对账活动远端 Task；
- Workflow/Team 的 pool、fanout、handoff、turn、budget 上限不因 A2A 绕过。

### 16.2 幂等和故障注入

- 响应前断网：请求可能已被接受但尚未拿到 taskId；
- 同 messageId 服务端去重和不去重两种实现；
- status/artifact 重复、乱序、分片重放；
- taskId/contextId 不匹配；
- Card 在运行中变化或 endpoint 漂移；
- 远端超时、卡死、取消竞态、终态后晚到事件；
- 远端错误声明 `COMPLETED` 但产物缺失/测试失败；
- Team Coordinator/进程重启后的恢复。

### 16.3 安全

- 恶意 Card description/skill prompt injection；
- 未信任 Card、错误证书、签名/信任锚不匹配；
- 跨 tenant task 查询/取消；
- 超大 text/base64/JSON、伪造 MIME、压缩炸弹；
- URL Part 指向 localhost、link-local、云 metadata、内网和 DNS rebinding；
- credential 泄漏到 Message、Artifact、日志或 LLM prompt；
- 远端请求 pool expansion、tool/model replacement、权限/预算提升；
- inbound 的认证绕过、速率限制和资源耗尽。

### 16.4 兼容性和性能

- 对试点 server 跑 [A2A TCK](https://github.com/a2aproject/a2a-tck) 的 MUST 集；
- 用 [A2A Inspector](https://github.com/a2aproject/a2a-inspector) 人工检查 Card、stream 和原始 payload；
- 记录端到端首 token、完成时间、序列化字节、重连恢复时长；
- 与现有本地 ACP adapter 做基线比较，明确 A2A 的价值来自远程互操作，而不是假设它更快。

## 17. 不建议做的事情

1. 不要把 `agent-client-protocol` 升级与 A2A 改造捆成一个大版本；两者分开评估、分开回滚。
2. 不要把每个 `tool+model` 组合自动暴露成公网 A2A Agent。
3. 不要让 LLM 读取任意 Agent Card 后直接加入 pool 或携带上游 credential 调用。
4. 不要用 A2A TaskStore 取代 Journal，也不要让两个状态机竞争 SSOT。
5. 不要把远端 `COMPLETED` 直接投影成本地 verified/finalized。
6. 不要假设 Send Message 天然 exactly-once；规范只给 MAY 幂等。
7. 不要首期做全互联 peer mesh、开放发现、push webhook 和自定义团队扩展。
8. 不要用 A2A 包装 provider 内部非权威、短命 subagent telemetry。
9. 不要在 Card/metadata 中暴露 token、系统 prompt、工作区绝对路径或完整内部 policy。

## 18. 最终建议

这个方向值得采用，但应把命题从：

> “把 Agent=LLM+Codex、subagent=LLM2+TraeX，然后都改成 A2A 通信”

改成：

> “把 GhostAP 的持久 Employee/受控远端团队成员建模为完整 A2A Agent；Employee 内部继续用 ACP 驱动冻结的 tool+model executor；Team/Workflow Kernel 通过一个受控 A2A Dispatch Adapter 与跨进程/跨主机 Agent 协作。”

这会更“丝滑”的地方是：

- Codex、TraeX 或第三方框架背后的 Agent 可以有统一远程契约；
- 长任务可以流式跟踪、断线查询/重订阅和取消；
- Agent 能公开能力但保持内部模型、记忆和工具不透明；
- Team 可以混合本地 ACP Employee 与远端 A2A Employee；
- 将来 GhostAP 可以作为标准 A2A Server 被其他系统调用。

它不会自动解决：

- Agent 创建、派生、预算、权限继承；
- 团队编排、循环控制、DAG、验证和最终完成判定；
- 工作区隔离、仓库锁和命令授权；
- exactly-once 执行；
- 本地调用延迟。

综合收益、改造风险与当前代码成熟度，推荐顺序是：**Employee outbound A2A 试点 → Team 受控远端成员 → Workflow remote pool member → 有真实外部调用需求后再做 inbound server → 最后才评估 push/扩展。**

## 19. 一手来源索引

- [A2A v1.0.1 README](https://github.com/a2aproject/A2A/blob/v1.0.1/README.md)
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/releases/tag/v1.0.1)
- [A2A Protocol Specification](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/specification.md)
- [Normative a2a.proto](https://github.com/a2aproject/A2A/blob/v1.0.1/specification/a2a.proto)
- [Key Concepts](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/key-concepts.md)
- [Life of a Task](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/life-of-a-task.md)
- [Agent Discovery](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/agent-discovery.md)
- [Streaming and Asynchronous Operations](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/streaming-and-async.md)
- [Extensions](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/extensions.md)
- [Extension and Binding Governance](https://github.com/a2aproject/A2A/blob/main/docs/topics/extension-and-binding-governance.md)
- [Enterprise Ready](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/enterprise-ready.md)
- [Multi-tenancy](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/multi-tenancy.md)
- [A2A and MCP](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/topics/a2a-and-mcp.md)
- [Official SDK index](https://github.com/a2aproject/A2A/blob/v1.0.1/docs/sdk/index.md)
- [a2a-python v1.1.2](https://github.com/a2aproject/a2a-python/blob/v1.1.2/README.md)
- [Official samples](https://github.com/a2aproject/a2a-samples/tree/6603ba3f2c31a7ef33e70b9d8b5b5f8be42ac9a3)
- [A2A TCK](https://github.com/a2aproject/a2a-tck)
- [A2A Inspector](https://github.com/a2aproject/a2a-inspector)
