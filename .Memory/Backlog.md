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

> **归档注释**：B020-B048 已按 `fixed`、`already satisfied`、`retired/superseded` 或 `external profile` 逐项记录处置依据；实现文件、精确测试/文档证据与保留边界见 [2026-07-16.md](2026-07-16.md)。强化多副本档的外部验收条件由 [employee runtime profiles ADR](../docs/adr-employee-runtime-profiles.md) 持续承载，不作为本地代码已证明能力。
