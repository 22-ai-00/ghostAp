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
| B052 | 2026-07-20 | 仓库级 Ruff 仍报告 85 条既有 Autonomous 测试告警（69 个 F401、4 个 F841、12 个 I001，分布在 31 个测试文件）；需在独立机械清理批次处理，避免与行为治理混杂。 | Low | 测试套件治理审计 | Open | — |
| B054 | 2026-07-22 | `lark-channel-sdk==1.1.0` 在 Python 3.13 导入时仍调用 protobuf `utcfromtimestamp()` 和无当前 loop 的 `asyncio.get_event_loop()`，产生两条上游 `DeprecationWarning`；关注 SDK 升级并在上游修复后移除兼容记录，不使用过滤器掩盖。 | Low | 普通编程 Channel 迁移 | Open | — |

| B061 | 2026-08-07 | 普通编程首轮耗尽约 6600 秒后虽进入预留收尾，但 provider 可把剩余约 600 秒全部用于上下文压缩，仍未返回最终答复。调整 reserve 会压缩主执行预算，单次日志不足以证明安全阈值；需增加 finalization 阶段耗时/事件观测并基于多次样本设计自适应收尾预算。 | Medium | 普通编程超时日志复盘 | Open | — |
| B062 | 2026-08-09 | Workflow 进度测试仍保留实现前的 RED 注释和 `object.__setattr__` 绕过，分页交付合同标题也仍标记为 RED；生产模型现已正式暴露对应字段。需在独立测试治理批次改用正常模型构造并校准文案，避免掩盖未来 schema 回归。 | Low | “继续执行”续接点调查 | Open | — |

> **归档注释**：B020-B048 已按 `fixed`、`already satisfied`、`retired/superseded` 或 `external profile` 逐项记录处置依据；实现文件、精确测试/文档证据与保留边界见 [2026-07-16.md](2026-07-16.md)。强化多副本档的外部验收条件由 [employee runtime profiles ADR](../docs/adr-employee-runtime-profiles.md) 持续承载，不作为本地代码已证明能力。

- 2026-08-10 [中/安全契约] `create_engine_session(require_tool_filter=True)` 当前只强制 ACP transport，不在工厂入口验证/安装具体 filter；现役 Workflow/Employee 调用方会随后安装 filter，但 API 容易被新调用方误用。后续应改为类型化 lane permission profile 或将 `tool_filter` 作为必填参数并在首个 prompt 前 fail-closed。
- 2026-08-10 [中/可靠性] `SPEC_EXECUTION_TIMEOUT` 当前被复用为各阶段 prompt timeout 与卡片 TTL，不是真正的 Spec 总运行 deadline；应增加 monotonic 总 SLA，并从其推导 phase timeout，避免自动任务无限循环或名称误导。

- 2026-08-10 [高/产品闭环] Slock 退役还遗留 `/fire`、`/history`、`/employee-memory` 三个 catalog 孤儿命令；当前 Journal services 仍在，应迁移为独立 EmployeeHandler 并恢复 admin/tenant/audit fail-closed 门禁。`/hire` 还缺 3ed4bcec 删除的 durable admission/start_hire，必须重建幂等、容量、唯一性、anchor-before-submit 与全自动默认工具模型流程，不能只补路由或复活旧 Slock UI。

## 员工 ACTION_REQUIRED 恢复入口（2026-08-10）

- 当前 `ACTION_REQUIRED` 员工的原因为 Slash Command reconciliation `recovery_exhausted`，现有控制面没有明确的重试/修复/重新授权转移路径。应补充可审计的恢复工作流，并恢复或替代当前孤立的 `/hire`、`/fire` 操作入口；职责标签更新不能替代生命周期恢复。

## Codex 子代理对账扩展（2026-08-10）

- 当前修复可从根父 rollout 补齐缺 path 的 direct child；若 ACP 同时缺失 nested child path，仍会安全地保持 incomplete。后续应使用已知 parent thread 的有界 rollout 闭包支持嵌套身份发现，并保持 parent/session/cwd/generation 一致性校验。
- `_rollout_candidates()` 仍按精确 session ID 在 `CODEX_HOME/sessions` 下执行目录遍历；真实事故三次定位约 39 ms，但历史持续增长后可能成为同步延迟。应引入 UUIDv7 日期分区定位或可信 session 索引，禁止退化为读取全历史 rollout 内容。
- 卡片 projector 与 ACP outcome 仍使用不同粒度的生命周期归并。后续应抽取共享、带 provenance/generation/malformed 语义的 reducer；在此之前不要让 outer `TOOL_CALL_DONE` 作为 child 权威终态进入最终 outcome。
