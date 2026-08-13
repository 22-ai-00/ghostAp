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
| B054 | 2026-07-22 | `lark-channel-sdk==1.2.0` 在 Python 3.13 导入时仍调用 protobuf `utcfromtimestamp()`，产生上游 `DeprecationWarning`；关注 SDK 升级并在上游修复后移除兼容记录，不使用过滤器掩盖。 | Low | 普通编程 Channel 迁移 | Open | — |

| B061 | 2026-08-07 | 单窗口耗尽后现已安全退休 transport、按原 provider session ID 自动恢复并续开新窗口，不再把一次 deadline 当作任务终态。仍需部署后采集真实租户 timeout → pause → retire → load_session → continue 的墙钟事件，校准 4 窗口默认值和 finalization reserve。 | Medium | 普通编程超时日志复盘 | Mitigated | — |
| B062 | 2026-08-09 | Workflow 进度测试仍保留实现前的 RED 注释和 `object.__setattr__` 绕过，分页交付合同标题也仍标记为 RED；生产模型现已正式暴露对应字段。需在独立测试治理批次改用正常模型构造并校准文案，避免掩盖未来 schema 回归。 | Low | “继续执行”续接点调查 | Open | — |
| B063 | 2026-08-11 | Workflow `retain_all_blocks` 会在每次进度刷新时深拷贝、序列化并重渲染完整执行历史；在 200 个直接调用与长输出下存在 O(total history) 内存/CPU 放大。需设计 append-only spool、页索引和仅渲染活动尾页，同时保持完整内容可追溯。 | Medium | Workflow 完整执行卡审查 | Open | — |
| B064 | 2026-08-11 | Workflow 的 `ACPStreamBridge` 尚未注入 handler 图片 uploader；Agent 截图/图片只会生成 `IMAGE_FAILED` 占位。需用窄 callback 接通上传，CardState 只保存 `image_key`，并补真实图片事件回归。 | Medium | Workflow 完整执行卡审查 | Open | — |
| B065 | 2026-08-11 | ACP `ToolCallProgress.raw_output` 为快照语义，但 reducer 仍按 delta 拼接；连续 `A`/`AB` 会形成重复或无效内容并可能 O(n²) 膨胀。需为 full Workflow 投影使用 replace/去重语义并覆盖多 progress 回归。 | Medium | Workflow 完整执行卡审查 | Open | — |
| B066 | 2026-08-11 | 完整 tool input/progress/output 固定使用三反引号；载荷自带 fenced Markdown 时会提前闭合并被当作卡片 Markdown 解释。需使用动态 fence 长度并让分页保持原 fence，覆盖跨页 fenced payload。 | Medium | Workflow 完整执行卡审查 | Open | — |
| B067 | 2026-08-11 | `agent-client-protocol==0.12.0` 仍在 Connection.close 时先停 dispatcher queue、后停 receive transport；2026-08-13 启动模型预热再次实证迟到帧会触发 `mssage queue already closed`。现由普通 session、通用/Coco 模型探测共享的 late-frame tolerant queue 兼容；关注上游修正后再移除适配。 | Low | ACP 关闭竞态 | Open | — |
| B068 | 2026-08-12 | Employee Ingress、Router、Outbox、dispatch history 和 Main-Bot warning 的 cursor 只避免“Journal 头未变”时重放；头被任何域推进后仍从 genesis 完整 replay，部分路径还解密全部历史 Blob、枚举 Blob 目录或周期扫描全部 terminal/pending 记录。Journal 增长后会放大空闲 worker、`/status`、恢复和关闭的 I/O/CPU。需实现可校验的增量 frame cursor、按域 pending/terminal 索引，并把全量 Blob 一致性检查移到启动或低频 reconciliation。 | Medium | 前六命令性能/稳定性复审 | Open | — |
| B069 | 2026-08-12 | Employee `/status` 的 `scoped_attempt_status()` 会在 Journal 头变化时于同步锁内重放 Gateway 投影，再全扫 attempts；Actor 状态仍读取 `queue.Queue.qsize()` 的近似值。需维护按 tenant/agent/chat/thread 的活动计数投影与锁保护 mailbox counter，使状态查询成本不随全域历史增长。 | Medium | 前六命令性能/稳定性复审 | Open | — |
| B070 | 2026-08-12 | `/stop`、`/history`、`/memory`、`/status` 及定向 `/task` 的确定性 Outbox 响应采用 `get_snapshot()` 后再 append；同一 acceptance 的并发首次调用可各自产生不同微秒 `created_at`，由第二个 writer 触发冲突而非稳定回读。需提供锁内 `get_or_create`/compare-and-return API，并覆盖每个 control 入口的多线程首次创建。 | Medium | 前六命令并发复审 | Open | — |
| B071 | 2026-08-12 | Fire 仍以全局生命周期互斥覆盖部分远端 effect，Membership 的每 chat 锁也覆盖 SDK 成员查询/变更；Hire 已把准入域锁与 submit 分离，但工具选择仍可同步探测 provider，完整 provisioning intent 没有统一 monotonic 总 deadline。慢 SDK/provider 会串行阻塞同域操作，且超时的线程调用仍可能迟到完成。需把远端 I/O 移出域锁后用代际/CAS 复核，按指定工具缩小探测并为整个 intent 建立可恢复 deadline。 | Medium | 前六命令性能/稳定性复审 | Open | — |
| B072 | 2026-08-12 | `test_ws_client_routing.py`（4,326 行/116 个测试函数）、`test_employee_team_gateway.py`（2,935/69）、`test_employee_router_queues.py`（2,488/44）、`test_employee_ingress_recovery.py`（1,696/36）与 `test_employee_membership_service.py`（1,023/34）已成为超大测试模块，增加定向运行、审查和冲突成本。需按 owner P2P、targeted group、handoff、terminal/reporting、membership recovery 拆分，同时保持共享 harness 单一来源。 | Low | 前六命令测试质量复审 | Open | — |
| B073 | 2026-08-12 | Targeted `/task` 已区分确定拒绝与 `INDETERMINATE`，但 READY transport、union identity、TrustZone/ACL、membership 与投影读取等依赖失败仍缺少分阶段、脱敏的 reason telemetry；现场只能看到最终 deny/unknown/retry 行为。需记录 dependency、stage、稳定 reason code 与延迟，禁止包含正文、Open ID、union ID 或凭据。 | Medium | 前六命令可观测性复审 | Open | — |
| B074 | 2026-08-12 | Dispatch reporting 的 `recovered_count=len(recovered)+reconciled` 可把同一 attempt 的“补终态”和“补快照”计为两次恢复；同时 missing/unreadable history 或 Outbox Blob 会保持周期 deferred，但缺少持久 ACTION_REQUIRED/运维修复入口。需拆分 attempt/snapshot 指标并为长期 poison record 建立可处置状态和告警。 | Medium | Employee reporting 恢复复审 | Open | — |
| B075 | 2026-08-12 | `EmployeeOutboxService.get_snapshot()` 以裸 `KeyError` 表示记录不存在，而 lifecycle 同时把该异常当作“首次创建”；投影/API 编程错误若也泄漏 `KeyError` 会被误归类为不存在。需增加类型化 `OutboxNotFoundError`，仅该异常允许进入创建分支。 | Medium | Employee Outbox 错误分类复审 | Open | — |
| B076 | 2026-08-12 | Main-Bot warning Outbox 已耐久化 Employee handoff/pre-start 终态告警，但普通消息 backpressure 与主 Bot identity 不可用仍有同步直发分支；warning `ACTION_REQUIRED` 也没有查询、重试或处置入口。需统一需要响应所有权的告警策略，并提供受审计的运维视图和重试动作。 | Medium | Main-Bot warning 交付复审 | Open | — |
| B077 | 2026-08-12 | Main-Bot warning 的公平轮转 cursor 与 delivery 锁只在单进程实例内；重启会从持久排序头重新开始，多副本也没有跨进程 delivery lease。其 origin digest 还以允许输入中出现的 NUL 作为字段分隔符。需持久化公平 cursor/claim lease，并改为长度前缀或 canonical JSON 编码坐标。 | Medium | Main-Bot warning 幂等/扩展性复审 | Open | — |
| B078 | 2026-08-12 | 启动 membership audit 现已在 runtime ready 前执行并失败关闭；但若旧版本留下“员工已 ARCHIVED、历史 ADD 未有因果更晚 REMOVE”的异常账本，而凭据已销毁，线上无法自动调用远端 API 证明清理。需提供离线 repair saga、人工证据锚定与明确运维手册，不能伪造 REMOVE 成功。 | Medium | Employee retirement 恢复复审 | Open | — |
| B079 | 2026-08-12 | `tests/conftest.py` 的 Node 泄漏诊断只在结束时枚举系统全部 Node 进程，没有记录测试启动基线，可能把用户或其他任务原有进程误报为本套件泄漏。需像线程诊断一样比较 PID/启动时间基线；当前没有证据表明本轮测试实际泄漏 Node。 | Low | 全量测试稳定性复审 | Open | — |
| B080 | 2026-08-12 | `AsyncCallbackBridge` 已为外部 callback task 加稳定命名并让 `drain()` 等待全部 sibling 后聚合失败，但仓库缺少直接覆盖“一个 sibling 取消/失败，另一个外部 callback 仍完成后才返回”的桥接回归。需增加独立 async contract test，避免后续改回 fail-fast gather。 | Low | 外部 mutation gate 覆盖复审 | Open | — |
| B081 | 2026-08-12 | `_sync_main_bot_identity()` 在进程内 identity lock 内执行同步飞书 SDK 请求；冷启动或缓存失效时，所有需要确认主 Bot 身份的入站消息会串行等待该调用，SDK 的迟到返回也不可取消。需后台 single-flight 预热、短调用 deadline 与 stale-safe 缓存，热路径只读快照并失败关闭。 | Medium | 主 Bot 入站延迟复审 | Open | — |
| B082 | 2026-08-12 | `EmployeeOutboxDeliveryCoordinator` 的 delivery lock 只在单对象内生效；若同一 Outbox 被两个 coordinator 实例共享，两者可同时取得同一 EXECUTING effect 并双重外发。第二个 commit 直接回读已有 binding，不核对自己的 receipt，因此可静默丢失先返回的 receipt。当前生产组装仅创建一个 coordinator，需将此单例拓扑固化为合同，或增加耐久 claim/lease 及 receipt 坐标 CAS，覆盖嵌入式/未来多副本部署。 | Medium | Employee Outbox 并发投递复审 | Open | — |
| B083 | 2026-08-12 | `FeishuWSClient.close()` 已在 SDK handler 创建前安装 binding barrier，但忽略 `WSHealthMonitor.disconnect()` 的 `False`；当 SDK disconnect 五秒超时或无可观测连接时，关闭仍可继续并返回成功。新 handler 会被 fence 拒绝，已调度 handler 也由 barrier 保护，因此未证明数据丢失；但底层 WS 资源仍可能滞留。需将 disconnect 结果纳入关闭结果/可重试资源所有权，并区分“本来无连接”与“断开超时”。 | Medium | WS 关闭 barrier 复审 | Open | — |
| B084 | 2026-08-12 | Scheduler completion callback 内若重入调用 `FeishuWSClient.close()`，`wait_for_completion_callbacks()` 会检测自身并返回未空，关闭随后进入 best-effort `scheduler.stop(wait=True)`；虽不会销毁 callback 依赖，但会经历自等待超时并返回 `False`。需显式拒绝 callback 内关闭或将关闭转移给独立 owner 线程，避免无效延迟与含糊返回值。 | Low | Scheduler completion 关闭复审 | Open | — |

| B085 | 2026-08-13 | Workflow 完整结果正文目前按 27 KiB 无损分页，超大模型输出可能产生数十张账本消息；需为用户可见卡片建立总页数/总字节上限，超量部分自动改由已生成的 HTML/Markdown 报告附件交付，同时保留摘要和恢复指引。 | Medium | Workflow 结果账本可读化审计 | Open | — |
| B086 | 2026-08-13 | Workflow 结果正文分页仍按 Unicode code point 选择切点，极窄边界可能把组合重音、肤色修饰符或 regional-indicator 字素拆到两页；需引入 grapheme-aware split boundary，并覆盖 emoji/组合字符/CJK wire 回归。 | Medium | Workflow 结果账本可读化审计 | Open | — |
| B087 | 2026-08-13 | 完整工具载荷净化虽已有 64 层/10000 节点上限，但高基数 opaque ID × 文本叶仍可能形成二次复杂度，净化后 mapping key 碰撞也可能覆盖先前值；需增加总字节/opaque 基数预算、线性多模式替换及稳定碰撞后缀。非 full-content JSON 解析也应统一捕获深层递归失败。 | Medium | Workflow 结果账本可读化审计 | Open | — |

> **归档注释**：B020-B048 已按 `fixed`、`already satisfied`、`retired/superseded` 或 `external profile` 逐项记录处置依据；实现文件、精确测试/文档证据与保留边界见 [2026-07-16.md](2026-07-16.md)。强化多副本档的外部验收条件由 [employee runtime profiles ADR](../docs/adr-employee-runtime-profiles.md) 持续承载，不作为本地代码已证明能力。

- 2026-08-10 [中/安全契约] `create_engine_session(require_tool_filter=True)` 当前只强制 ACP transport，不在工厂入口验证/安装具体 filter；现役 Workflow/Employee 调用方会随后安装 filter，但 API 容易被新调用方误用。后续应改为类型化 lane permission profile 或将 `tool_filter` 作为必填参数并在首个 prompt 前 fail-closed。
- 2026-08-10 [中/可靠性] `SPEC_EXECUTION_TIMEOUT` 当前被复用为各阶段 prompt timeout 与卡片 TTL，不是真正的 Spec 总运行 deadline；应增加 monotonic 总 SLA，并从其推导 phase timeout，避免自动任务无限循环或名称误导。

## 员工 ACTION_REQUIRED 恢复入口（2026-08-10）

- 当前 `ACTION_REQUIRED` 员工的原因为 Slash Command reconciliation `recovery_exhausted`，现有控制面没有明确的重试/修复/重新授权转移路径。应补充可审计的恢复工作流，并恢复或替代当前孤立的 `/hire`、`/fire` 操作入口；职责标签更新不能替代生命周期恢复。

## Codex 子代理对账扩展（2026-08-10）

- 当前修复可从根父 rollout 补齐缺 path 的 direct child；若 ACP 同时缺失 nested child path，仍会安全地保持 incomplete。后续应使用已知 parent thread 的有界 rollout 闭包支持嵌套身份发现，并保持 parent/session/cwd/generation 一致性校验。
- `_rollout_candidates()` 仍按精确 session ID 在 `CODEX_HOME/sessions` 下执行目录遍历；真实事故三次定位约 39 ms，但历史持续增长后可能成为同步延迟。应引入 UUIDv7 日期分区定位或可信 session 索引，禁止退化为读取全历史 rollout 内容。
- 卡片 projector 与 ACP outcome 仍使用不同粒度的生命周期归并。后续应抽取共享、带 provenance/generation/malformed 语义的 reducer；在此之前不要让 outer `TOOL_CALL_DONE` 作为 child 权威终态进入最终 outcome。
