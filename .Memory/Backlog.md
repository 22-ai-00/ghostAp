# Maintenance Backlog

> **用途**：收集 Low/Medium severity 的审计缺口，集中在维护窗口批量处理，避免打断主线开发节奏。
>
> **工作流**：Review/Audit 产出的 gap 按分级标准评估 → High 立即修复 → Low/Medium 录入本表 → 每两周维护窗口集中处理。

## 分级标准

| Severity | 定义 | 处理方式 |
|----------|------|----------|
| **High** | 影响正确性、安全性、数据丢失 | 立即修复，不入 Backlog |
| **Medium** | 可观测性、可运维性缺口（如日志错误、配置缺失） | 录入 Backlog，维护窗口处理 |
| **Low** | 代码风格、文档一致性、命名规范 | 录入 Backlog，维护窗口处理 |

## Backlog 条目

| ID | 日期 | Gap 描述 | Severity | 来源 | 状态 | 解决 Commit |
|----|------|----------|----------|------|------|-------------|
| B049 | 2026-07-16 | Feishu API 硬超时后 daemon SDK worker 无法取消；本地删除 binding 后，迟到 PATCH 可能越过新代际远端写入。需设计 request generation/远端见证并做故障注入。 | Medium | Deep 卡片顺序分页审计 | Open | — |
| B051 | 2026-07-16 | 员工 Contact/Context/群历史 SDK 调用缺少 endpoint、员工 app、message_id、平台错误码与分段耗时关联；异常目前多被压缩为 false/unknown，现场只能结合 Journal 推断。需补脱敏结构化观测。 | Medium | Team 员工延迟日志审计 | Open | — |
| B052 | 2026-07-20 | 仓库级 Ruff 仍报告 96 条既有 Autonomous 测试告警（未使用导入、局部变量与 import 排序）；需在独立机械清理批次处理，避免与行为治理混杂。 | Low | 测试套件治理审计 | Open | — |
| B053 | 2026-07-20 | 快速层仍有少量 2–4 秒 retry/集成测试依赖真实等待；优先用 fake clock/Event 消除等待，确属真实进程/时间契约的迁入 `slow`。 | Low | 测试套件治理审计 | Open | — |
| B054 | 2026-07-22 | `lark-channel-sdk==1.1.0` 在 Python 3.13 导入时仍调用 protobuf `utcfromtimestamp()` 和无当前 loop 的 `asyncio.get_event_loop()`，产生两条上游 `DeprecationWarning`；关注 SDK 升级并在上游修复后移除兼容记录，不使用过滤器掩盖。 | Low | 普通编程 Channel 迁移 | Open | — |
| B055 | 2026-07-24 | `TaskOrchestrator` 已兼容未知标题的 `kind="agent"` 与未知标题、`kind="other"`、含 `子代理：` 的旧 provider 事件，但缺少分别隔离这两个正向分支的直接路由回归；在测试维护批次补参数化用例，防止后续分类重构退化。 | Low | Deep 假子任务路由终审 | Open | — |
| B056 | 2026-07-30 | 普通编程续接分页采用 append-only 历史快照；若子代理任务简述晚于所属页面冻结，既有标题会保留“子任务”而无法回填。需在来源注册与首个文本帧之间定义冻结前标签门禁，或明确接受历史快照语义。 | Low | 普通编程分段卡片终审 | Open | — |
| B057 | 2026-07-31 | `SelectionFlowController.snapshot()` 未持久化 `error_message`；finish 空选择时设置的错误会在重新构造 controller 后丢失，可能导致内联提示不显示。状态机仍会拒绝空选择，后续补快照字段与恢复回归。 | Low | Workflow 卡片洪泛终审 | Open | — |
| B058 | 2026-07-31 | Autonomous employee dispatch 空闲轮询每秒无条件重建 ingress/router projection 并扫描 BlobStore；当前约 2200 Journal 帧、328 blobs/140MB 时持续占用约 0.6 CPU 核。需用 anchor/cursor/queue 变化检测跳过无变化重放，并降低 GC 扫描频率。 | Medium | 服务无响应排障 | Open | — |
| B059 | 2026-08-05 | `SlockHandler._has_slock_permission()` 标注返回 `bool`，但配置管理员且 sender 为空时会因布尔表达式返回空字符串；当前调用方均按 truthiness 使用且保持拒绝，后续应显式 `bool(...)` 收紧类型合同。 | Low | 测试资产深度精简审计 | Open | — |
| B060 | 2026-08-05 | `FeishuCardAPIClient` 为 CardKit 实体路由新增的进程内 ID 集合尚无有界生命周期，`card.update` 的 sequence conflict / 普通运输失败也缺直接错误映射回归；后续以有界路由注册表取代集合并补参数化故障注入。 | Low | Deep 消息洪泛终审 | Open | — |

| B061 | 2026-08-07 | 普通编程首轮耗尽约 6600 秒后虽进入预留收尾，但 provider 可把剩余约 600 秒全部用于上下文压缩，仍未返回最终答复。调整 reserve 会压缩主执行预算，单次日志不足以证明安全阈值；需增加 finalization 阶段耗时/事件观测并基于多次样本设计自适应收尾预算。 | Medium | 普通编程超时日志复盘 | Open | — |
| B062 | 2026-08-07 | SandboxExecutor 日志曾发现仓库由聊天 A 持锁却为聊天 B 执行的告警，发生于另一仓库且未证明与本次卡片问题相关；需定位漏加 repo lock 的 handler 并补跨聊天隔离回归。 | Medium | 普通编程日志复盘 | Open | — |

> **归档注释**：B020-B048 已按 `fixed`、`already satisfied`、`retired/superseded` 或 `external profile` 逐项记录处置依据；实现文件、精确测试/文档证据与保留边界见 [2026-07-16.md](2026-07-16.md)。强化多副本档的外部验收条件由 [employee runtime profiles ADR](../docs/adr-employee-runtime-profiles.md) 持续承载，不作为本地代码已证明能力。
