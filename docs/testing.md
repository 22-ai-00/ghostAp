# 测试治理

GhostAP 的测试目标是用最小维护面保护真实产品契约，而不是追求用例数量。新增或保留测试时，先问它能否捕获一个 GhostAP 实现错误；只验证 Python、dataclass、Enum 或 mock 本身的测试没有独立价值。

## 运行层级

```bash
# 修改后先跑最相关文件或 node id
uv run python -m pytest tests/test_target.py -q

# 日常广覆盖反馈；排除真实等待、进程和故障注入慢测
uv run python -m pytest tests/ -q -m "not slow"

# 慢速契约；与上面的快测合起来等价于全量
uv run python -m pytest tests/ -q -m slow

# 发布、合并前的单命令全量验证
uv run python -m pytest tests/ -q

# 只读清单：数量、代码行、最大文件和完全同体测试候选
uv run python scripts/test_inventory.py tests/
```

`slow` 不是跳过标签。验证真实超时、跨进程故障、断线重放、ACK 边界或外部运行时的测试仍是发布门禁，只是不应阻塞每次本地快速反馈。预计超过 1 秒且确实依赖真实等待/进程的测试应标记 `slow`；能用 Event、假时钟或注入超时值稳定验证的，优先消除真实等待。

## Workflow 真实租户多卡验收

`scripts/validate_workflow_tenant.py` 是只读、显式 opt-in 的证据校验器；它不会连接飞书或发送消息。真实验收必须在专用测试 chat 中走完整 `/wf` 交互，且任务应产生至少两个结果页。原始 tenant/chat/message ID 不得写入证据文件，只记录对应的小写 SHA-256 摘要。

先用本次运行的非机密绑定创建默认失败的 `0600` 检查清单；文件采用独占创建，已存在时会拒绝覆盖：

```bash
uv run python scripts/validate_workflow_tenant.py \
  --template-out /tmp/workflow-live-capture.json \
  --run-id <run-id> \
  --service-instance-id <service-instance-id> \
  --tenant-hash <sha256> \
  --chat-id-hash <sha256> \
  --expected-result-count <count>
```

执行真实 `/wf` 后填写以下证据，任何默认值都不能通过：

- `events` 按实际发生顺序记录 `create`、`patch`、`freeze`、`finish`，序号连续；每页始终使用同一个 `message_id_hash`，每次载荷记录 `payload_sha256`。
- 新续页的 `create` 必须早于旧页 `freeze`；旧页冻结后不得再记录 `patch`；最后仅最新页记录 `finish`。
- `checks` 逐项确认 origin/recipient 绑定、历史页冻结后哈希不变、全部终态结果可见、停止按钮消失、桌面端和移动端连续展示。
- `artifacts` 记录脱敏事件日志、桌面截图和移动截图的 SHA-256；`attestor` 记录验收人，`observed_result_count` 必须等于绑定的预期数量。

最后使用完全相同的绑定显式打开 live 门禁；退出码 `0/1/2` 分别表示通过、失败、缺少绑定或尚未验收：

```bash
GHOSTAP_WORKFLOW_ACCEPTANCE_LIVE=1 \
uv run python scripts/validate_workflow_tenant.py \
  --live \
  --live-results /tmp/workflow-live-capture.json \
  --run-id <run-id> \
  --service-instance-id <service-instance-id> \
  --tenant-hash <sha256> \
  --chat-id-hash <sha256> \
  --expected-result-count <count>
```

## 执行通道纵向合同

`tests/contracts/test_direct_programming_lane.py` 从显式 Slash 命令或当前 programming mode 一直覆盖到真实 session-manager/factory 请求。测试侧 recorder 只替换进程/远端传输边界，并记录 backend、实际 factory model、cwd、chat/project/thread session key、tool filter 与 prompt；它不以 mocks 重建路由。显式 Direct 请求的准入是一个目标 factory 加一个目标 prompt，且没有 classifier、planner、reviewer 或 coordinator 调用。

`tests/contracts/test_protected_execution_lanes.py` 固定 Deep/Spec 的 provider/model、
同 session 的真实首尝试超时后重试、factory 与 process-local pause/resume 路径。
它不宣称 durable recovery，也不要求两条引擎改为同一内部算法。

运行 `uv run python scripts/benchmark_direct_lane.py --runs 20` 会输出 Direct 远端调用拓扑分布。它只拒绝多出的 factory/prompt hop，不设置墙钟毫秒阈值；本地调度、CPU 与缓存差异不应改变该合同。Claude CLI 的 Direct factory/prompt 记录必须保留所选 model；`[1m]` 选择在真实进程 argv 中使用基础 model，并在真实进程环境中携带对应 Anthropic beta。

## 准入与删减规则

| 类别 | 决策 | 例子 |
| --- | --- | --- |
| 安全、权限、并发、锁、Journal/effect、持久化恢复 | 必须保留 | fail-close、fsync/anchor、重放、竞态终态 |
| 外部协议与用户可见 schema | 必须保留 | 飞书 Card JSON、ACP/CLI 传输、稳定枚举值 |
| 同一行为在不同边界可捕获不同故障 | 分层保留 | 纯 reducer 契约 + 一条真实投递集成 |
| 同一函数、同一路径、同一断言的重复测试 | 只保留最强且最接近契约的一条 | 两个文件重复检查同一按钮集合 |
| 大量同规则输入 | 按等价类合并 | greeting/ack/task 各保留大小写、语言和边界代表值 |
| dataclass 构造器回显、相等性、多个字段的 frozen 重复检查 | 删除 | `Model(x=1).x == 1`、分别检查 int/bool 都不可赋值 |
| Enum/Python 自带语义 | 删除 | 成员彼此不等、字符串值本身是字符串 |
| 稳定序列化值、向后兼容默认值、mutable default 隔离 | 合并后保留 | Journal 状态值、缺字段读取、独立 list/dict 默认值 |
| 复制生产映射或常量文本再断言副本 | 删除 | 在测试里重建 renderer mapping 后检查测试自己的 key |
| public re-export/import 边界 | 仅在兼容性是明确产品契约时保留 | 对外模块迁移兼容；普通内部 import smoke 不保留 |

## 编写方式

- 回归测试名称应描述曾经或可能发生的失败，不使用波次、任务编号或“exists/works”替代行为说明。
- 一个实现错误只需要一个最小回归；集成测试只有在能捕获单元测试捕获不到的接线、序列化或生命周期错误时才重复覆盖。
- 参数样例先按等价类收敛。大量廉价输入可以在单条契约内循环，并在断言消息中包含失败样例；边界条件或不同控制流仍拆开。
- 禁止为降低数字删除高风险契约，也禁止把失败、异常退出或缺失摘要当作通过。
- `scripts/test_inventory.py` 只报告候选。不同 backend/engine 的同体测试可能保护独立实现，删除前必须确认调用目标和故障检测能力相同。

## 审计节奏

功能开发仍按“相关测试 → 扩大子系统 → 快测 → 慢测/全量”执行。测试文件超过约 1,000 行、单文件超过约 60 个测试函数、出现新的完全同体候选，或全量耗时明显上升时，安排一次局部治理；不设置强制删除比例。
