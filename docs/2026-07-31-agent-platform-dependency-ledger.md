# Agent Platform Evolution Dependency Ledger

> 原审计基线：`dev@8d15f26ba6874ab9e19f6972bad7b30dd953a540`
>
> 本次方案复核：`dev@3d581ead41ae0c98d16630fc2c5009f8223c4714`
>
> 审计/修订日期：2026-07-31
>
> 主计划：`docs/2026-07-31-agent-platform-evolution-plan.md`
>
> Foundation：`docs/2026-07-30-ghostap-product-convergence-plan.md`

本账本是两份计划的去重、依赖和验收事实源。本次修订只调整方案，没有把计划状态误写成代码已实现。

## Status and disposition

状态只按当前主计划的完整验收口径计算：

- `missing`：当前合同、生产接线或验收证据不存在。
- `partial`：存在可复用底座，但没有形成当前合同要求的生产闭环。
- `complete`：当前目标测试、生产接线、恢复/失败路径和文档证据全部通过。
- `complete-old`：旧合同已完成，但该合同已被新易用性模型替换；不能作为新检查点通过证据。

任务处置单独记录：

- `active`：属于核心剩余路径并阻塞某个检查点。
- `folded`：由另一个 canonical work package 唯一实现，不再重复建设。
- `evidence-only`：实现已完成，只保留回归证据。
- `deferred`：出现真实需求后再做，不阻塞核心体验。
- `superseded`：旧设计明确删除，不应继续实现。

相邻测试通过不等于任务完成；外部真实飞书检查只能记录为 `passed`、`failed` 或 `not_tested`，不得用模拟结果冒充。

## 2026-07-31 ease-first decision

当前产品是单 Owner、单主机、受信工程环境。最高优先级是 Owner 在普通编程和 GhostAP 自建 Agent 群中零权限打断。

Canonical trust model：

1. `OWNER_P2P`：唯一 Owner 私聊。可改项目授权、Backend、凭据、全局设置和 Host Shell；Owner 明确命令直接执行，不二次确认。
2. `MANAGED_AGENT_GROUP`：GhostAP/Employee 创建、Registry 持久登记并绑定 project/team 的群。Owner 和有效 Employee continuation 在项目内直接工作，不查 legacy enrollment、不逐 Run approval。
3. `EXTERNAL_OR_UNKNOWN_GROUP`：不执行；未知成员不创建 Run、Ledger、context 或 session。

删除的首版机制：

- tenant 业务授权、每对象 ACL、ACL revision；
- `AuthorityEnvelope`、`DataEgressPolicy`、RiskClass；
- 逐 Run approval、standing order、nonce/digest、approval inbox；
- sandbox/network egress conformance 作为 proactive 启用门槛；
- 签名插件市场、多租户发布、8/50 Employee 容量门、完整 Eval 治理。

静默保留的正确性机制：

- Journal SSOT、PREPARED/EXECUTING 双锚定；
- idempotency/dedupe、真实 cancel acknowledgment、unknown reconciliation；
- Run limits/deadline、automation kill、restart recovery、backup/restore；
- Vault/secret redaction、scope/provenance、Direct/Deep/Spec 保护合同。

## Baseline facts

- 原隔离环境非慢速基线：
  `12664 passed, 9 skipped, 118 deselected, 45 subtests passed`。
- Phase 0/A 原相邻回归：`143 passed`。
- Phase B 原相邻回归：`202 passed`。
- Foundation 原相邻回归：`24 passed`。
- 已完成且保留证据的任务：Foundation F1–F3（其中 F3 为旧合同）和
  Phase 0 的 0.4–0.7。
- 新三信任域、ManagedGroup Registry、ingress cutover、ProjectGrant/ActionMatrix
  尚无实现证据；因此 CP-T 未通过。

## Canonical ownership and overlap decisions

| 重叠范围 | 唯一实现归属 | 兼容/删除策略 |
| --- | --- | --- |
| Trust zone、Actor、ActionMatrix | `src/trust/models.py`、`resolver.py`、`action_matrix.py` | legacy user/chat allowlist 只留迁移诊断；矩阵无 `ASK` |
| 受管群 provenance 与 ProjectGrant | `src/trust/registry.py` | Project/Slock marker、群名、allowed chat 不能单独证明 trust |
| Project/Team 建群生命周期 | ProjectChatService/Slock handler → Trust Registry | ACTIVE 早于 welcome；失败不报成功 |
| Ingress/callback trust cutover | `src/feishu/ws_client.py` + immutable route context | trust 判定早于 Ledger/image/project/classifier/session/Effect |
| Foundation 9 / 0.1 产品动作与执行通道 | `src/feishu/product_catalog.py` | 旧 Slash 列表只从 catalog 投影 |
| Foundation 13 / B1 任务读控 | `src/tasking/` | ingress 已解析 trust；内部对象不重复 Principal ACL |
| Foundation 14 / A1–A3 Backend 与 Session | `src/agent_session/` catalog/models/drivers | 旧 factory 只保留委托 facade |
| Foundation 15–16 / A6 路由 | `src/agent_session/routing.py` | 显式选择永远优先；provider-local retry |
| Foundation 17 / B2、B5、B6 Run 生命周期 | 现役 `src/autonomous/` Journal/domain/scheduler + 窄 `src/orchestration/` facade | 不平行重写第二套内核 |
| Foundation 18 / 0.5 / B8 Worktree | Worktree adapter + orchestration black-box node | 0.5 证据保留；本地 merge 可自动，外部动作按 ActionMatrix |
| Foundation 20 / 0.4 / B4/B5/B10 Workflow | IR v2 + Autonomous durable ports | v1 只作紧急回退，不保留第二运行事实 |
| Foundation 22 / E1 doctor | `src/diagnostics/readiness.py` | process-fallback 是 trusted-host 信息，不全局 degraded |
| Foundation 23 / E3 backup | `src/operations/backup.py` | 覆盖 Journal/anchor/Vault/Blob/workspace/trust registry |
| Foundation 24 sandbox | optional hardening backlog | 不阻塞 managed-project Agent 或 proactive |

原 Foundation Task 25 不实施。Task 26 只保留本机启用、回退和验证；不实现 tenant beta、签名晋级或观察窗口。

## Canonical work packages and critical path

| 优先级 | Work package | 包含/折叠 | 依赖 | 剩余人日 | 检查点 |
| --- | --- | --- | --- | ---: | --- |
| P0 | Trusted Project + Protected Lanes | 0.1–0.3、0.8–0.10；0.4–0.7 为 evidence-only | F1–F3 旧基线 | 7–11 | CP-T、CP-P0 |
| P1 | Built-in Backend Contract | A1–A6；F14–F16 折入 | CP-P0 | 8–13 | CP-A |
| P1 | Compile Domain | B1–B4、B9a；F13/F17 折入 | CP-P0；可与 Backend 并行 | 7–11 | CP-B-Compile |
| P1 | Durable Execute Kernel | B5/B6/B10；复用 Autonomous 内核 | Compile + Backend | 9–14 | CP-B-Execute |
| P2 | Workflow/Team/Worktree adapters | B7；B8 按需，B9b deferred | CP-B-Execute | 3–5 | extension checks |
| P2 | Useful Proactive Runtime | C1–C8 首版 | CP-T + CP-B-Execute | 10–16 | CP-C-Managed |
| P3 | Context + Operations | D4、E1、E3、E5 | Artifact/Journal 稳定 | 7–12 | 持续验收 |
|  | **核心剩余** |  |  | **51–82** |  |

A7、B9b、cron/DST、daily digest/quiet hours、D1–D3/D5、8/50 负载与多副本均为 deferred，不属于上表核心估算。

## Foundation disposition

| Task | 状态 | 处置 | 当前验收/说明 |
| --- | --- | --- | --- |
| F1 公共产品合同 | complete (`4b998c09`) | evidence-only | 文档合同 6 passed；README/metadata/命令一致 |
| F2 类型化安全姿态 | complete (`4b998c09`) | folded | 配置类型证据保留；权限产品模型由 CP-T 取代 |
| F3 入站默认拒绝与引导 | complete-old (`b3e30b4f`) | superseded | user∧chat enrollment 已完成但会阻断自建群；不能证明 CP-T |
| F4 Host Shell 显式授权 | missing | folded → 0.8 | raw Host Shell 仅 Owner P2P；Backend 项目工具为 trusted-host profile |
| F5 Employee 显式启用 | partial | folded → C8 | 只保留 emergency off；Trust Registry 不依赖 Employee enablement |
| F6 群上下文作用域 | missing | folded → 0.10/D4 | ACTIVE managed group + project/thread scope + secret 过滤 |
| F7 topic/mode 转换 | partial | active → CP-P0 | 编程入口优先、完整别名、stale callback revision |
| F8 onboarding 真相 | missing | active → 0.1/E5 | 文案/命令迁移，无退休入口漂移 |
| F9 ProductAction catalog | complete | folded → 0.1 | `src/feishu/product_catalog.py` 提供单一 Owner 执行通道事实源 |
| F10 菜单/帮助 | missing | folded → 0.1 | trust zone/project 信息；删除角色权限矩阵 |
| F11 EffectiveContext | partial | folded → 0.8/0.10 | trust 解析一次、immutable context、dispatch 静默 revision check |
| F12 RouteDecision/Executor | partial | folded → 0.8/0.10 | deep-freeze、reason enum、唯一副作用执行器 |
| F13 统一任务读控 | partial | folded → B1 | 全引擎 adapter、幂等 stop/retry；无对象 ACL |
| F14 Backend catalog/session request | partial | folded → A1–A5 | trust/profile/binding 合同 |
| F15 provider/mode 漂移 | missing | folded → A6 | 持久 binding，显式失败不 fallback |
| F16 Spec provider-local retry | missing | folded → A6 | provider-local candidates/fingerprint |
| F17 共享 run/checkpoint | missing | folded → B1–B6 | 不单独实现第二生命周期 |
| F18 Worktree timeout/terminal | partial | evidence-only + B8 optional | 0.5 已完成；black-box adapter 另算 |
| F19 Deep progress recovery | missing | deferred | 首版只诚实标 `recoverable=False` 且不重复派发 |
| F20 Workflow truthful recovery | partial | folded → B4/B5/B10 | IR v2 + durable ports |
| F21 退休 Autonomous 隔离 | partial | folded → B5/C8 | legacy writer/consumer 禁用 |
| F22 readiness doctor | missing | folded → E1 | trust registry、Journal、run、trigger、kill |
| F23 备份恢复 | missing | folded → E3 | 加入 trust registry/project grants |
| F24 OS Shell isolation | missing | deferred | optional hardening；不阻塞核心路径 |

## Phase 0 disposition

| Task | 状态 | 处置 | 硬验收 |
| --- | --- | --- | --- |
| 0.1 执行通道产品合同 | complete | evidence-only | maturity/health/visibility；Owner 无完成度门禁，显式保护命令不被 Slock 截获 |
| 0.2 Direct 纵向合同 | missing | active | 单目标 prompt、零 planner hop、真实 cancel/retry/session |
| 0.3 Claude CLI 模型真实性 | partial | active | argv/env 真正绑定 model/1M；失败不污染选择 |
| 0.4 Workflow binding/reviewer | complete (`ffeb2e82`) | evidence-only | immutable RunSpec、真实 binding/reviewer；Workflow 全集 945 passed |
| 0.5 Worktree 终态/超时/评审 | complete (`9054d221`) | evidence-only | hard timeout、cancel ack、证据 review、无部分自动 merge |
| 0.6 Spec completion fail-closed | complete (`93a52598`) | evidence-only | 明确失败不能被空建议转成通过 |
| 0.7 辅助 Agent 权限 | complete (`f69acc90`) | evidence-only | coordinator/classifier deny-all；不产生用户交互 |
| 0.8 TrustZone/Actor/ActionMatrix | missing | active | `ALLOW/DENY` only；Owner/Employee/unknown + stale revision |
| 0.9 ManagedGroup Registry/lifecycle | missing | active | ACTIVE 早于 welcome；Project/Team 共用；tombstone/replay |
| 0.10 ingress/callback cutover | missing | active | trust 早于业务副作用；managed group 零 enrollment；external 零副作用 |

CP-T 与 CP-P0 必须在 A/B/C 的生产 cutover 前通过。

### Task 0.1 evidence

- Production wiring: `src/feishu/product_catalog.py` defines the only local
  execution-lane completion/health metadata; the `/menu` card renders its
  Owner surface; `src/feishu/dispatcher.py` uses the protected-command
  projection before Slock handling; both Slock welcome templates no longer
  advertise retired `/goal`.
- TDD RED: `uv run pytest tests/test_product_action_catalog.py -q` produced
  `1 failed, 5 passed`; the expected failure showed Slock detection captured
  explicit `/codex` instead of calling the system route.
- GREEN: the focused catalog contract was `7 passed`; adjacent menu, catalog,
  welcome-card, WebSocket routing, and document regressions were `150 passed`.
  Touched-file Ruff and `git diff --check` passed.
- The completion label only annotates product support.  All implemented lane
  entries remain Owner-visible and directly accessible; no allowlist, rollout,
  approval, sandbox, or release-state gate was added.
- Real Feishu/manual evidence: `not_tested`.  No local test result is recorded
  as a tenant or manual delivery pass.  This task does not change recovery,
  cancel, unknown-effect, or permission-prompt behavior; their runtime/manual
  evidence remains outside this narrow catalog task.

## Phase A disposition

| Task | 状态 | 处置 | 当前硬验收 |
| --- | --- | --- | --- |
| A1 中立能力与 SessionRequest | missing | active | binding/purpose/trust zone/project root/profile 类型化 |
| A2 唯一 Backend Catalog | missing | active | family/binding/effective capabilities 单一事实源 |
| A3 Driver 统一工厂 | partial | active | Direct 拓扑不变，旧入口仅委托 |
| A4 discovery/mutation 分离 | partial | active | 探测不安装/更新；freshness/degraded 可见 |
| A5 Conformance Kit | missing | active | start/send/close/resume/cancel/events；aux deny-all；无 sandbox gate |
| A6 确定性 Routing | missing | active | 无 LLM；显式选择优先；失败不跨 provider fallback |
| A7 Owner custom binding | missing | deferred | Owner P2P + argv + 现有 driver；不建签名插件系统 |

## Phase B disposition

| Task | 状态 | 处置 | 当前硬验收 |
| --- | --- | --- | --- |
| B1 Run Read/Control | partial | active | 所有引擎统一 RunView/stop/retry；trust 在 ingress；无 tenant/object ACL |
| B2 冻结版本化 IR | partial | active | schema/migration/零副作用 compile；ActionKind 由 adapter 确认 |
| B3 Artifact/Provenance/Done | partial | active | `scope_ref`、hash/lineage、机器可验 done；secret 不进 Artifact |
| B4 Workflow IR v2 编译 | missing | active | 单一生成源、诊断、零 dispatch、v1 fallback |
| B5 耐久生产端口 | partial | active | ActionMatrix 后 PREPARED/EXECUTING 均先 fsync+anchor |
| B6 Scheduler/Limits/Cancellation | partial | active | 单机 generation、Run deadline/limits、真实 cancel ack |
| B7 Team/Slock 事实源收敛 | partial | active | Team durable graph；Slock 不另建执行队列 |
| B8 Worktree black-box node | missing | deferred/按需 | local merge true gates；外部动作按 ActionMatrix |
| B9a 真实综合 | partial | active | reviewer 结果和冲突证据进入 synthesis |
| B9b 有界 PlanPatch | missing | deferred | 不阻塞静态 DAG；Coordinator 永不扩 grant/limits/criteria |
| B10 恢复/卡片/运维视图 | partial | active | schema migration、restart reconcile、current activity、无 approval UI |

CP-B-Compile 通过前 IR v2 零外发；CP-B-Execute 通过前 IR 不驱动真实 Effect。通过后直接对 Owner 开放，不增加审批阶段。

## Phase C disposition

| Task | 状态 | 处置 | 当前硬验收 |
| --- | --- | --- | --- |
| C1 Goal/Trigger/ProjectGrant | partial | active | frozen durable projection；无 AuthorityEnvelope/DataEgressPolicy |
| C2 one-shot/interval/misfire | missing | active | injectable clock、稳定 occurrence、有界 catch-up；cron deferred |
| C3 adapters/Occurrence ingress | missing | active | transport verified + trust/grant revision；adapter 无执行权 |
| C4 admission/dedupe/generation/DLQ | partial | active | occurrence-bound 与 run-created 原子提交 |
| C5 shared ActionMatrix/effect gate | missing | active | managed project 允许直执；拒绝只摘要，无 approval |
| C6 limits/stop/automation kill | partial | active | Journal-backed kill、per-run stop、一次 resume |
| C7 cards/notification coalescing | missing | active | 先做 `ux/`；保留现有折叠框；无 risk/approval card |
| C8 production composition | missing | active | `disabled|managed_project|connected`、恢复顺序、emergency off |

CP-C-Managed 的关键证据：

- managed project read/write/Shell/test/build/local Git/编排无 approval node；
- duplicate occurrence = 0，kill anchor 后新 dispatch = 0；
- connected target 外的自动外部写 = 0；
- opaque unknown 不自动 retry，只阻塞所属 Run；
- Direct/Deep/Spec 合同不变。

## Phase D/E disposition

| Task | 状态 | 处置 | 当前硬验收 |
| --- | --- | --- | --- |
| D1 Execution Trace | missing | deferred | 最小 trace 字段可先折入 B events；隐藏 CoT/secret 不落盘 |
| D2 offline eval/replay | missing | deferred | 出现真实 routing 优化需求后做 |
| D3 CapabilityProfile | missing | deferred | 无 tenant；project 仅统计分组 |
| D4 ContextEnvelope-lite | partial | active | scope/freshness/token/provenance；跨项目默认不选；secret 过滤 |
| D5 evidence Auto routing | missing | deferred | 显式选择优先；external mutable Effect 不 exploration |
| E1 readiness/metrics | partial | active | trust/Journal/Vault/run/trigger/kill；process fallback 信息化 |
| E2 workload baseline | missing | deferred | 只保留 1 Direct + 1 Team + 1 Trigger restart/cancel smoke |
| E3 backup/restore | missing | active | point-in-time restore；trust registry + grants；unknown 不重放 |
| E4 direct enable/fallback | partial | folded → 0.10/C8 | 完成即启用；每子系统一个 emergency switch |
| E5 Owner E2E checklist | missing | active | managed group 零提示、external 零执行、explicit action 不二次确认 |

## Completion record requirements

每完成一个 `active` 任务，必须在本账本将状态改成 `complete` 并附上：

- 红测名称和失败原因；
- 绿测、相邻回归和静态检查；
- production wiring 位置；
- 恢复/取消/unknown/失败路径证据；
- `permission_prompt_count(managed project task) = 0` 证据；
- 尚不能本地执行的真实飞书项的 `not_tested` 记录。

被标为 `deferred` 或 `superseded` 的任务不得继续作为 CP-T、CP-B 或 CP-C 的隐含前置。
