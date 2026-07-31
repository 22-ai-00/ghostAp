# Agent Platform Evolution Dependency Ledger

> 基线：`dev@8d15f26ba6874ab9e19f6972bad7b30dd953a540`
>
> 审计日期：2026-07-31
>
> 主计划：`docs/2026-07-31-agent-platform-evolution-plan.md`
>
> Foundation：`docs/2026-07-30-ghostap-product-convergence-plan.md`

本账本是两份计划的去重、依赖和验收事实源。状态只按计划的完整验收口径计算：

- `missing`：计划要求的合同、生产接线或验收证据不存在。
- `partial`：存在可复用底座，但没有形成计划要求的生产闭环。
- `complete`：目标测试、生产接线、恢复/失败路径和文档证据全部通过。
- 相邻测试通过不等于任务完成；外部真实飞书检查只能记录为
  `passed`、`failed` 或 `not_tested`，不得用模拟结果冒充。

## Baseline

- 工作树在审计前为空。
- 隔离环境外非慢速基线：
  `12664 passed, 9 skipped, 118 deselected, 45 subtests passed`。
- Phase 0/A 相邻回归：`143 passed`。
- Phase B 相邻回归：`202 passed`。
- Foundation 相邻回归：`24 passed`；其中旧授权测试仍固化
  “空 allowlist 等于 allow-all”，不能作为新合同证据。
- 当前没有任何任务满足计划的完整完成定义。

## Canonical ownership and overlap decisions

| 重叠范围 | 唯一实现归属 | 兼容策略 |
| --- | --- | --- |
| Foundation 9 / 0.1 产品动作与执行通道 | `src/feishu/product_catalog.py` | 旧 Slash 列表只从 catalog 投影，不再维护第二份集合 |
| Foundation 13 / B1 任务读控 | `src/tasking/` | 引擎 adapter 实现统一协议；旧诊断入口委托控制面 |
| Foundation 14 / A1–A3 Backend 与 Session | `src/agent_session/catalog.py`、`models.py`、`drivers/` | `backend_catalog.py`、旧 factory 仅保留委托 facade |
| Foundation 15 / A6 默认选择与路由 | `src/agent_session/routing.py` | 持久选择保存完整 binding；显式选择失败关闭 |
| Foundation 16 / A6 Spec retry | provider-local discovery/routing | 禁止借用 Coco model manager 或跨 provider fallback |
| Foundation 17 / B2、B5、B6 Run 生命周期 | `src/orchestration/` + `src/tasking/` | 各引擎状态通过 adapter 映射，不复制生命周期枚举 |
| Foundation 18 / 0.5 / B8 Worktree | Worktree adapter + orchestration black-box node | 一次修复 timeout、cancel ack、终态、评审和 merge gate |
| Foundation 20 / 0.4 / B4、B5、B10 Workflow | IR v2 + durable orchestration journal | v1 只作为紧急回退，不保留第二份运行事实 |
| Foundation 22 / E1 doctor | `src/diagnostics/readiness.py` | 删除旧签名/租户晋级语义，保留本机结构化检查 |
| Foundation 23 / E3 backup | `src/operations/backup.py` | 一个 manifest 覆盖 Journal/anchor/Vault/Blob/workspace/config |
| Foundation 24 / E4 Shell isolation | `src/sandbox/` | 可选 OS profile 必须失败关闭；Host Shell 仍需显式授权 |

原 Foundation Task 25 不实施。Task 26 仅保留本机启用、回退和验证；不实现
tenant beta、签名晋级或 48 小时发布门。

## Foundation Track status

| Task | 状态 | 直接依赖 | 完成证据 |
| --- | --- | --- | --- |
| F1 公共产品合同 | missing | 无 | 文档合同测试；README/metadata/命令叙述一致 |
| F2 类型化安全姿态 | missing | F1 | posture 单测、配置校验、结构化输出 |
| F3 入站默认拒绝与引导 | partial | F2 | user/chat 双维 deny-by-default、原子持久化、并发测试 |
| F4 Host Shell 显式授权 | missing | F2、F3 | P2P/管理员/聊天授权矩阵与拒绝回归 |
| F5 Employee 显式启用 | partial | F2 | enable flag 默认关闭且早于 Vault/Journal 构造 |
| F6 群上下文作用域与过期 | missing | F3、F5 | membership/consent/retention/tombstone/restart 测试 |
| F7 topic/mode 转换正确性 | partial | F1 | 编程入口优先、完整别名、stale callback fencing |
| F8 onboarding 真相 | missing | F1、F7 | 文案/命令迁移测试，无 `/goal`、`--prompt` 漂移 |
| F9 ProductAction catalog | missing | F1、F7 | 单一 catalog、Owner surface、兼容别名合同 |
| F10 角色感知菜单/帮助 | missing | F9、F11 | `ux/` 预览、桌面/移动结构测试、权限视图 |
| F11 EffectiveContext | partial | F7 | resolve-once、不可变上下文、跨入口一致性测试 |
| F12 RouteDecision/Executor | partial | F9、F11 | deep-freeze、reason enum、唯一副作用执行器 |
| F13 统一任务读控 | partial | F12 | 全引擎 adapter、作用域再授权、幂等 stop/retry |
| F14 Backend catalog/session request | partial | F2、F12 | 由 A1–A5 共同验收 |
| F15 provider/mode 漂移 | missing | F14 | 由 A6 与持久 binding 合同共同验收 |
| F16 Spec provider-local retry | missing | F14、F15 | provider-local candidates/fingerprint 回归 |
| F17 共享 run/checkpoint | missing | F13、F14 | 由 B1–B6 共同验收 |
| F18 Worktree timeout/terminal | partial | F17 | 由 0.5/B8 共同验收 |
| F19 Deep progress recovery | missing | F17 | checkpoint/restart/reconcile、无重复派发 |
| F20 Workflow truthful recovery | partial | F17 | 由 0.4/B4/B5/B10 共同验收 |
| F21 退休 Autonomous 隔离 | partial | F12 | legacy import boundary、中性 presentation |
| F22 readiness doctor | missing | F13–F21 | 由 E1 验收 |
| F23 加密备份恢复 | missing | F17、F21 | 由 E3 验收 |
| F24 可选 OS Shell isolation | missing | F4 | fail-closed backend、进程树取消、环境最小化 |

## Phase 0 status

| Task | 状态 | Foundation/后继关系 | 硬验收 |
| --- | --- | --- | --- |
| 0.1 执行通道产品合同 | missing | 合并 F9；保护 A/B 迁移边界 | maturity/health/visibility 合同，无退休入口广告 |
| 0.2 Direct 纵向合同 | missing | A3 cutover 前置 | 单目标 prompt、零额外 LLM hop、真实 cancel/retry/session |
| 0.3 Claude CLI 模型真实性 | partial | F14/F15/A3 | argv/env 真正绑定 model/1M；失败不污染选择 |
| 0.4 Workflow binding/reviewer | partial | B4/B9a 前置 | immutable RunSpec、真实 binding、真实 reviewer call |
| 0.5 Worktree 终态/超时/评审 | partial | 合并 F18/B8 | hard wall clock、session cancel ack、无部分自动合并 |
| 0.6 Spec completion fail-closed | partial | 保护 Spec | FAIL/非 JSON/无证据均拒绝完成 |
| 0.7 辅助 Agent 权限 | missing | A1/A5 安全前置 | classifier/coordinator/summarizer deny-all 工具合同 |

CP-P0 必须在任何成熟路径迁移前通过。

## Phase A status

| Task | 状态 | 依赖 | 硬验收 |
| --- | --- | --- | --- |
| A1 中立能力与 SessionRequest | missing | F2、0.2、0.7 | binding/purpose/security/environment 均类型化 |
| A2 唯一 Backend Catalog | missing | A1 | family/binding/effective capabilities 单一事实源 |
| A3 Driver 统一工厂 | partial | A1、A2、CP-P0 | Direct 拓扑不变，旧入口仅委托 |
| A4 discovery/mutation 分离 | partial | A2、0.7 | 探测不安装/更新，freshness/degraded 可见 |
| A5 Conformance Kit | missing | A1–A4 | 每 Backend 对权限、取消、usage、egress 声明给证据 |
| A6 确定性 Routing | missing | A2、A5、F15/F16 | 无 LLM，解释 reason/considered，显式失败不 fallback |
| A7 受信扩展边界 | missing | A2–A6 | 签名 manifest/schema/registry，聊天入口不可安装 |

## Phase B status

| Task | 状态 | 依赖 | 硬验收 |
| --- | --- | --- | --- |
| B1 Run Read/Control | partial | F13、A1 | 所有引擎统一 RunView/stop/retry/ACL |
| B2 冻结版本化 IR | partial | B1 | schema/version/migration/无副作用 compile |
| B3 Artifact/Provenance/Done | partial | B2 | ArtifactRef、ACL、lineage、机器可验 done criteria |
| B4 Workflow IR v2 编译 | missing | B2、B3、0.4 | 单一生成源、诊断、零 dispatch、v1 fallback |
| B5 耐久生产端口 | partial | B2–B4 | PREPARED 与 EXECUTING 均先 fsync+anchor 再外发 |
| B6 Scheduler/Budget/CancellationScope | partial | B5 | durable lease/fence、资源所有权、真实 cancel ack |
| B7 Team/Slock 事实源收敛 | partial | B1、B5、B6 | Team 单一 durable graph；Slock 不另建执行队列 |
| B8 Worktree black-box node | missing | 0.5、B3、B5、B6 | 隔离输出、review artifact、显式 merge approval |
| B9a 真实综合 | partial | B3–B6、0.4 | 独立 reviewer 结果和冲突证据进入 synthesis |
| B9b 有界 PlanPatch | missing | B9a、B6 | 次数/预算/权限有界，patch 可审计 |
| B10 恢复/卡片/运维视图 | partial | B5–B9 | schema migration、restart reconcile、current activity |

CP-B-Compile 通过前 IR v2 零外发；CP-B-Execute 通过前 IR 不驱动写操作。

## Phase C status

| Task | 状态 | 依赖 | 硬验收 |
| --- | --- | --- | --- |
| C1 Goal/Trigger/Authority | partial | B2、B5 | frozen durable projections + DataEgressPolicy |
| C2 clock/cron/misfire | missing | C1 | injectable clock、DST/回拨、有界 catch-up |
| C3 adapters/Occurrence ingress | missing | C1、C2 | 统一持久 ingress，adapter 无执行权 |
| C4 admission/dedupe/lease/DLQ | partial | C3、B5/B6 | occurrence-bound 与 run-created 原子提交 |
| C5 共享 policy/effect gate | missing | C4、CP-B-Execute | 与手工任务共用 orchestration/policy/effect ports |
| C6 approval/budget/kill | partial | C5 | Journal-backed kill、nonce/scope/TTL/budget 绑定 |
| C7 cards/digest/quiet hours | missing | C4–C6 | 先做 `ux/` 预览；节流、暂停、run-now、异常升级 |
| C8 production composition | missing | C1–C7 | replay 顺序、反向关闭、`PROACTIVE_ENABLED` 紧急开关 |

CP-C-Assist 只允许只读；CP-C-Write 必须逐 Run Owner 确认。

## Phase D status

| Task | 状态 | 依赖 | 硬验收 |
| --- | --- | --- | --- |
| D1 Execution Trace | missing | CP-B-Execute | 无隐藏思维链；binding/version/usage/artifact lineage |
| D2 offline eval/replay | missing | D1 | fake adapter、dataset schema、确定性 replay |
| D3 证据化 CapabilityProfile | missing | D1、D2 | verifier evidence、版本/租户作用域；不改员工身份 |
| D4 ContextEnvelope | partial | B3、D1 | item provenance/freshness/sensitivity、summary artifact |
| D5 证据推荐/Auto routing | missing | A6、D3、D4 | 能力硬过滤、reason/confidence、安全探索 |

## Phase E status

| Task | 状态 | 依赖 | 硬验收 |
| --- | --- | --- | --- |
| E1 readiness/metrics | partial | A–D、F22 | catalog/Journal/Vault/Blob/lag/run/trigger/kill/isolation |
| E2 Owner workload baseline | missing | CP-B-Execute、CP-C-Assist | 8 Employee、restart/card/context/trigger workload |
| E3 backup/restore/rollback | missing | F23、D4 | deletion high-water/revocation，隔离恢复证明 |
| E4 direct enable/fallback | partial | E1–E3 | 默认 Owner 可见、子系统级紧急开关，无 rollout 遗留 |
| E5 Owner E2E checklist | missing | E1–E4 | 自动 smoke + 手工项结构化三态，不伪报 |

## Serial critical path and parallel work

1. **事实与安全止血**：F1–F8，随后 0.7、0.6、0.5、0.3、0.4。
2. **保护合同**：F9–F12 与 0.1/0.2；通过 CP-P0。
3. **连接平台**：合并实现 F14–F16 与 A1–A6；A7 在静态 catalog
   稳定后实施；通过 CP-A。
4. **控制与耐久执行**：F13/F17–F20 与 B1–B6/B9a/B10；
   B7/B8/B9b 可在共享合同稳定后并行；依次通过 CP-B-Compile/Execute。
5. **主动闭环**：C1–C8；先 CP-C-Assist，再 CP-C-Write。
6. **证据与上下文**：D1–D5。
7. **运维收口**：F21–F24 与 E1–E5；CP-E 为持续清单，不是发布门。

每完成一个任务，必须在本表将状态改成 `complete` 并附上：

- 红测名称和失败原因；
- 绿测、相邻回归和静态检查；
- 生产接线位置；
- 恢复/取消/失败关闭证据；
- 尚不能在本地执行的真实飞书项的 `not_tested` 记录。
