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
| B054 | 2026-07-22 | `lark-channel-sdk==1.2.0` 在 Python 3.13 导入时仍调用 protobuf `utcfromtimestamp()`，产生上游 `DeprecationWarning`；2026-11-19 实测 PyPI `lark-channel-sdk==1.4.0` wheel 仍在 `lark_channel/ws/pb/google/protobuf/internal/well_known_types.py:91` 调用 `datetime.datetime.utcfromtimestamp(0)`，故保持锁定 `1.2.0` 且不使用过滤器掩盖，待上游移除后再升级。 | Low | 普通编程 Channel 迁移 | Open | — |
| B061 | 2026-08-07 | 单窗口耗尽后现已安全退休 transport、按原 provider session ID 自动恢复并续开新窗口，不再把一次 deadline 当作任务终态。仍需部署后采集真实租户 timeout → pause → retire → load_session → continue 的墙钟事件，校准 4 窗口默认值和 finalization reserve。 | Medium | 普通编程超时日志复盘 | Mitigated | — |
| B067 | 2026-08-11 | `agent-client-protocol==0.12.0` 仍在 Connection.close 时先停 dispatcher queue、后停 receive transport；2026-11-19 实测 `agent-client-protocol==1.0.0` 已移除 `InMemoryMessageQueue` / `Connection(..., queue=...)` 而 `ToolCallStart` 仍要求 `title: str`，故当前需继续锁定 `0.12.0` 并保留 `LateFrameTolerantMessageQueue`，同时承载关闭迟到帧兼容与缺失 `title` 的部分 `tool_call` 帧改写。 | Low | ACP 关闭竞态 | Open | — |
| B069 | 2026-11-24 | `src/workflow_engine/run_spec.py:140-159`：冷缓存下 `WorkflowRunSpec.from_dict()` 当 `binding.model_name is None` 且 `(tool, profile, effort)` 一致时信任 `binding_specs[aid]["effective_model"]` 的基础模型名（与 `data["agent_pool"]` 同级信任，可在冷重载校验边界进一步收紧）。 | Low | Workflow M1 评审 | Open | — |

> **归档注释**：B020-B048 已按 `fixed`、`already satisfied`、`retired/superseded` 或 `external profile` 逐项记录处置依据；实现文件、精确测试/文档证据与保留边界见 [2026-07-16.md](2026-07-16.md)。
