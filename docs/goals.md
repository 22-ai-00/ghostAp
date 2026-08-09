# GhostAP 长期目标与历史执行上下文

> 当前公开产品定位以
> [`docs/product-contract.md`](./product-contract.md) 为唯一权威：GhostAP 是面向
> 受信工程环境的飞书/Lark 原生 Agent Department Gateway。主 Bot 是控制面，
> Employee Bot 是独立执行身份；provider/transport 可替换；Host Shell 是必须关闭
> 或显式授权的特权宿主执行能力，不是操作系统沙箱。内置员工档案仅承诺单机、
> 文件持久化，不承诺多副本线性一致性或对特权宿主机的回滚抵抗。

以下内容保留长期目标和历史执行证据。继续开发时不要从头重做，也不要把局部模块、
自动化测试或安全脚手架误认为整个目标已经完成。

  工作目录：
  /data00/home/jiataorui/work/github/ghostAp

  原始完整需求：
  /data00/home/jiataorui/.codex/attachments/3121b5aa-316a-41ca-bee6-6374a07ca4a0/pasted-text-1.txt

  首先必须完整读取：

  1. 原始需求附件
  2. /data00/home/jiataorui/work/github/ghostAp/AGENTS.md
  3. .Memory/Abstract.md
  4. .Memory/2026-07-12.md
  5. .Memory/2026-07-13.md
  6. docs/2026-07-12-autonomous-agent-department-design.md
  7. docs/2026-07-12-autonomous-foundation-plan.md
  8. docs/2026-07-12-autonomous-data-plan.md
  9. docs/2026-07-13-autonomous-hire-production-plan.md
  10. docs/2026-07-13-autonomous-thread-context-plan.md
  11. docs/2026-07-14-autonomous-employee-response-plan.md
  12. docs/2026-07-14-autonomous-membership-stop-plan.md
  13. docs/2026-07-14-autonomous-fire-plan.md

  然后以当前 Git、源码、运行配置和新鲜测试结果为唯一权威状态进行检查：

  cd /data00/home/jiataorui/work/github/ghostAp
  git status --short --branch
  git log -15 --oneline
  git rev-parse HEAD
  git rev-parse origin/dev
  cat .superpowers/sdd/progress.md 2>/dev/null || true
  rg --files .superpowers/sdd 2>/dev/null || true

  重要：用户已经明确授权直接在 dev 分支开发，不要创建 worktree。保留用户已有改动，不要 reset、checkout 或删除任何不属于你的修改。

  当前权威状态（2026-07-14；必须用 Git、源码、配置和测试重新核验，不可只相信此摘要）：

  - 本文档不再固化会在下一次提交立即过期的 HEAD；每次继续开发必须以上述
    `git rev-parse HEAD/origin/dev` 的现场结果为准。Thread Context Task 2 开发前
    二者均位于 `f83585a504de3a372fa9a29106739efec7230393`。
  - Foundation 已完成并保留：
    - Canonical frozen Employee/BotPrincipal domain
    - AES-GCM Credential Vault
    - Journal-backed employee/Bot/alias/authority projection
    - 安全可重建 identity.json
    - Tenant-aware ProjectedAgentRegistry
    - Legacy/V5 authority cutover 与持久化失败恢复
  - `/hire` 的生产形态代码已经实现，不再只是内存脚手架：
    - 官方 `lark-oapi==1.7.1` 一键创建应用 adapter 与精确 manifest
    - Journal/Vault-backed durable Hire Saga 与 callback bridge
    - PREPARED/EXECUTING 锚定、幂等恢复与 ACTION_REQUIRED 语义
    - 每员工 fresh-interpreter Channel 子进程、一次性 secret pipe、generation fencing 和 sandbox attestation
    - Slash GET/diff/POST/PATCH/DELETE/GET 精确 reconciliation
    - 真实员工 Bot `/status`、nonce、主 Bot send-count=0 的激活验证
    - `EmployeeDepartmentRuntime` 生产 composition、恢复 supervisor 和 FeishuWSClient 生命周期接线
    - 首次启动自动生成本地 Journal/Vault/Data 密钥并使用本地 FileAnchor
  - 旧的独立 Autonomous Manager 命令面（`/goal`、`/goals`、`/run`、`/runs`、
    `/approve`、`/approvals`、`/decisions`）不再作为生产入口，已明确退役并 fail-close；
    旧目标命令已接入消息调度器。
  - `/hire` 管理员 DM 卡片权限 Bug 已修复：消息事件入口保存官方 `event.message.chat_type` 与 origin/chat/operator；卡片回调不再读取不存在的 `context.chat_type`。只有服务端明确无 provenance 记录时才查询 Chat API，并且只读结构字段 `chat_mode`；API 结果必须原子写回完整可信绑定。来源查询/写入失败、残缺、过期、跨 chat、跨 operator 或并发冲突均 fail-close。
  - readiness 反馈已接入处理器：只有 provider 明确返回 `ready=True` 且无 blockers 才派发真实 Hire；否则向管理员显示具体安全门禁，不再误报“不是管理员”，也不降级创建本地虚拟角色。
  - Visible Employee 已改为内置启动能力：默认 `autonomous_visible_employee_limit=8`，不再需要 root-owned release broker、evidence/attestation、release binding 或人工 sandbox 标记。首次启动在 `AUTONOMOUS_STATE_DIR` 下创建 mode `0600` 的本地密钥；显式设为 `0` 仍可完全关闭员工 runtime。
  - 真实员工闭环阶段状态：
    - Thread Context Task 1-6 已完成并关闭 Phase 2：冻结 contracts/config、employee-scoped 官方
      `lark-oapi` message source、每员工 Vault credential lease、Get current 权威
      root/thread 解析、Thread 双遍历稳定 snapshot、Group 双窗口 recent、current
      propagation、revision/identity 去重、watermark、整单元预算裁剪、稳定错误与关闭
      线性化，以及 tenant-bound canonical L1、membership/chat-bound full L2、动态
      Projected Registry/BotPrincipal、requester/current-sender 绑定、mandatory atomic
      authority fence 均有回归；production runtime 现拥有 Context source/service 与
      canonical Data composition，拆分 hire/execution readiness，并覆盖 employee app probe、
      shared Journal 同步、restart/rotation/retirement invalidation 与逆序关闭；真实页间
      insert/edit/delete、timeout、重复 token、partial SDK、两把密钥轮换、restart 与 shutdown
      故障注入均证明 `CONTEXT_UNAVAILABLE` 零 task/ACP 派发；Phase 3 已把 durable ingress 接入该服务
    - Phase 3 Task 0-7 已完成 durable employee ingress、Projected Registry/ACL/membership
      authority、Journal-backed bounded Router、锚定 dispatch attempt、Context gate、
      本地九 selector 已精确聚合；员工 runtime 现在由本地安全自举直接启用，真实租户证据只用于验收，
      不再作为启动门禁
    - Phase 4 Employee Response Channel 已完成：冻结 snapshot/binding/effect、员工密钥加密 Blob、
      Journal replay、稳定 UUIDv5 单卡 create、employee child public `update_card` patch、四元组回执栅栏、
      terminal fencing、恢复 worker 与 runtime ownership 均已接线；旧 in-memory provisioning response 已删除，
      delivery coordinator 不持有主 Bot 端口，因此不存在 fallback 路径
    - Phase 5 已完成真实团队 membership 与 `/stop` 唯一终态；Bot 成员关系只由
      `member_id_type=app_id` 变更和目标员工凭据 `is_in_chat` 观察确认，未知结果默认拒绝
    - Phase 6 已完成生产 `/fire`：legacy `provisioning/fire.py` 仅保留兼容测试，新路径使用 shared
      Journal 的 RETIRING/Effects/ACTION_REQUIRED/ARCHIVED 状态机、durable 单员工 ingress closure、
      verified Slash cleanup、Channel stop、remove-only membership cleanup、Vault destroy 与原子归档
    - Phase 7 已完成数据面真实切换：独立 canonical authority 在 legacy import 及验证后单向锚定；
      completed attempt 在员工结果卡前幂等发布 L1、chat/thread summary、skill profile 与 task reasoning，
      restart 可补齐文档而不重跑 ACP；read ACL 只信任 transport 与 workforce 投影，完整 L1 仅管理员主
      Bot DM 可读；迁移拒绝 symlink/冲突/损坏，恢复校验全部 live Blob 并重建投影文件
    - Phase 8 已重构为内置员工启动：本地私有密钥原子生成并跨重启复用；完整显式密钥仍可使用，
      但禁止显式与自动配置混用。主 Bot 本地审计属于员工 runtime 的完整性门禁；打开、校验或
      记录失败会 fail-close 员工 mutation。该审计仍是单机本地证据，不等同于强化多副本档的
      外部不可回滚 witness。
    - 员工 Channel 优先使用 bwrap；宿主 user namespace 不可用时会销毁失败子进程及其 pipe，随后
      用同一 `python -I`、一次性 secret pipe 和最小环境协议重试一次，并明确记录
      `verified=false, mechanism=process-fallback`，不会伪报强隔离。
  - 尚无获授权的真实测试/生产租户执行证据。本地测试证明代码合同，但不替代真实 Bot、双租户、
    桌面/移动 Slash、主 Bot 零代发和 1/10/50 Bot soak。发起用户仍需打开注册链接并按飞书官方指引
    完成 Bot 应用创建；该交互步骤不要求租户管理员审批，也不是部署门禁。

  完整最终目标：

  构建可以投入生产的 GhostAP Autonomous “Agent Department”：

  - 每个员工是独立飞书 Bot，拥有自己的 app_id/app_secret、名字、头像、Slash Commands。
  - 使用一键创建应用 SDK 完成 `/hire` Provisioning Saga。
  - 每员工使用独立 Channel SDK WebSocket。
  - 支持任务提交查询链路、`/status`、`/history`、`/memory`、`/stop`。
  - 支持 `/fire` 与团队 membership 管理。
  - 执行历史按日 JSONL，但 Journal + encrypted Blob 才是事实源。
  - L1、skill、reasoning 使用 Journal-first 数据面。
  - Context 顺序严格为：
    Thread 全量消息 > 群最近消息 > L1 > L2。
  - 员工必须用自己的 Bot 输出状态卡和结果，绝不 fallback 到主 Bot。
  - 不引入数据库，继续使用文件存储。
  - 最后完成自动化、故障注入和真实飞书租户验收，并 push 到 dev。

  绝对不能破坏：

  - Deep / Spec / Workflow 引擎逻辑和路由。
  - 主 GhostAP Bot 现有 WebSocket 连接和消息入口。
  - SMART、普通编程模式和 topic-scoped engine 状态合同。
  - Journal SSOT、frozen domain、Effect dispatch 前锚定、默认拒绝策略。
  - 不得把 app_secret 写入 identity.json、Journal、日志、卡片、异常、argv、环境变量或普通 IPC。

  开始任何新阶段前：

  - 阅读该阶段对应设计、计划和最近 Memory 证据。
  - 使用 `rg` 检查现有 domain、Vault、BlobStore、Settings 和相邻生产模式。
  - 确认当前分支、HEAD、origin/dev 和工作树状态。
  - 先运行最相关的基线；修改共享路由、卡片、锁、配置、会话或启动 composition 时扩大回归。

  推荐基线：

  uv sync --group dev
  uv run python -m pytest tests/autonomous/ -q
  uv run python -m pytest tests/test_docs_references.py -q
  uv run python -m src.main --validate
  git diff --check

  如果基线失败：

  - 先判断是现有失败、环境问题还是当前未提交改动。
  - 使用 systematic-debugging 查明根因。
  - 不得无视与当前任务相关的失败继续开发。

  从当前 HEAD 继续的实施顺序：

  1. 已完成（2026-07-13）修复 `/hire` 卡片权限上下文：
     - 不再读取 SDK 模型中不存在的 `context.chat_type`
     - 使用服务端保存、绑定 origin message/chat/operator 的可信 DM provenance
     - provenance 查询失败、残缺、过期、跨 chat、跨 operator 或冲突时 fail-close；仅明确无记录可用 Chat API `chat_mode` 原子回填
     - readiness 未满足时展示具体安全阻断，不误报“不是管理员”
  2. 已完成（Task 1-6）Thread Context 生产接线：
     - 已完成：冻结 scope/revision/watermark/config contracts
     - 已完成：employee-scoped 官方 FeishuMessageSource、root_id → thread_id
       权威解析、严格全量分页 primitive、revision/edit/delete/content normalization
     - 已完成：Thread 双遍历稳定 snapshot、current propagation、Group recent 双窗口与
       cutoff cohort、watermark/source boundary、跨层 identity/revision 去重
     - 已完成：trusted constraints reserve + protected current，以及
       L2 → L1 → group recent → oldest Thread 整单元裁剪
     - 已完成：tenant/projected-owner canonical L1、membership/chat-bound full L2、
       ACTIVE/visible/principal/app/credential/generation/requester ACL、current sender
       绑定、`CONTEXT_UNAVAILABLE` 零执行和 mandatory atomic authority-fence contract
     - 已完成：Data projection rebuild/read/publish/GC 统一锁所有权，以及 canonical
       L1 root/parent/final dir-fd no-follow containment
     - 已完成：Task 5 production composition、employee-scoped execution readiness、
       recovery/rotation/retirement invalidation、shared Journal coordination 与 shutdown ownership
     - 已完成：Task 6 真实页间 mutation、deadline/token/partial SDK、restart/rotation/shutdown
       failure injection；所有 mandatory Context failure 均零 delegate/task/ACP 派发，三路终审批准
     - Phase 2 handoff 已关闭；下一阶段是 durable employee ingress
     - 设计与现有实现审计已完成，实施计划见
       `docs/2026-07-13-autonomous-durable-ingress-plan.md`；Task 0-5 的 durable Inbox、
       official Channel ACK bridge、附件暂存、Journal-backed Router、Task 6 真实执行与
       Task 7 生产组合/恢复/本地证据聚合均已完成；Phase 3 已关闭
     - 已完成 Task 0 SDK capability 硬门禁：锁定 `lark-channel-sdk==1.1.0` 的
       low-level message 与 P2 `card.action.trigger` EVENT 均通过真实 HTTP/WSS/protobuf
       ACK 顺序验证；高层 handler 提前 200 与 raw `MessageType.CARD` 继续明确禁用
     - standalone runner 冻结 19 个 wire selectors，绑定 Git/pyproject/uv.lock wheel/
       RECORD/runtime payload/受控空 bytecode cache；mismatch 稳定阻断为
       `employee_channel_sdk_capability_mismatch`，本地 evidence 永远不可晋级
     - 已确认 SDK 的单帧 byte limit 缺口：超大单帧仍会进入 callback，因此 evidence
       强制 `requires_parent_payload_gate=true`；Task 3 已用真实 wire 证明 parent gate
       返回 non-success 且 Inbox/Journal/Router/ACP 零副作用，Task 7 已纳入九 selector 聚合
     - 已完成 Task 1：六个 frozen exact-schema ingress 模型、递归 secret alias 拒绝、
       restart-safe canonical dedup、1.5 秒 ACK/attachment/payload 配置、单员工≤团队≤全局
       队列关系，以及独立 Phase 3 implementation evidence manifest 均已冻结
     - 已完成 Task 2：独立 AES-GCM BlobStore + Journal-backed Inbox、anchor 后 ACK、
       并发/restart/generation dedup、语义/provenance 冲突拒绝、缺失/损坏 payload
       恢复关闭、orphan quarantine、可重试 tombstone GC 与 commit 前 disposition 校验
     - `EI-IPC-01` 已由真实 spawn child + Pipe + FileAnchor + Journal/fsync selector
       收集，观测 ACK `0.014952s <= 1.5s`；Task 7 已将全局 FI-29 严格桥接到该 selector，
       但当前没有外部 final-build attestation，FI-29 实际状态仍为 PENDING，不构成生产 readiness
     - Task 2 独立 Spec/Code review 已批准
     - 已完成 Task 3：生产 worker 仅使用官方 low-level WSClient/dispatcher；message 与
       real P2 card callback 在同一 1.5 秒总 deadline 内等待 parent durable ACK，
       `EI-BRIDGE-MESSAGE-01`/`EI-BRIDGE-CARD-01` 真实 WSS/protobuf selectors 已通过
     - Task 3 mandatory matrix 已覆盖 IPC backpressure/partial frame、parent close、
       anchor/projection/ACK encode/control write、late/lost ACK、child crash、SDK write、
       reconnect/STOP/generation rotation；connection epoch 与 READY/INGRESS 顺序 fail-close
     - Task 3 独立 Spec/Code review 已批准；该阶段当时仍把外部 release trust 作为门禁。
       这一历史状态已被 2026-07-14 的内置员工决策取代；当前默认 limit 为 8，严格外部
       trust 属于 hardened multi-replica profile，见 `docs/adr-employee-runtime-profiles.md`
     - 已纠正 Channel ACK 假设：高层 `FeishuChannel` 消息回调会先 schedule 后返回，
       不能证明平台 ACK 发生在 Journal fsync/anchor 之后；实现必须通过锁定版本的
       low-level dispatcher 黑盒验证消息和 CardAction 两条路径，任一路径不满足即
       保持 execution readiness 关闭
     - 已完成：Channel ACK 前 Journal durable Inbox
     - 已完成 Task 4：ACK 路径只加密 typed resource descriptor、不下载附件；授权后仅使用
       employee credential 与官方 `lark-oapi` message-resource API，无 Manager Bot fallback
     - 已完成 Task 4：0700/0600、dir-fd/no-follow、server-random name、parent/leaf durable
       identity、count/size/timeout/MIME+magic/executable/hash/hardlink/generation 校验，以及
       Gateway-only trusted path export
     - 已完成 Task 4：cleanup 在 aggregate completion 前对 exact bound inode 执行
       `ftruncate(0)+fsync` 并 fresh-reopen 全量复核；完成后不执行存在 TOCTOU 的 pathname
       unlink，`cleanup_completed` 保证敏感字节已持久擦除但不承诺删除零字节目录项
     - Task 4 最终 70 focused、332 expanded、1479 full Autonomous 测试通过，独立
       Spec/Code review 均批准且无 Critical/Important
     - 已完成 Task 5：删除旧内存 production Router；durable Router 只消费 anchored Inbox，
       绑定 ACTIVE visible employee/tenant/Bot/app/精确 READY enum/generation/connection/
       membership/requester ACL/workforce coordinate，并持久化完整生命周期与有界 FIFO
     - 已完成 Task 5：team/global fairness rebalance 在单帧原子终止过度占用的 queued victim
       并接纳 newcomer；final workforce fence、三次附件 authority/credential 校验、terminal-only
       Task 4 cleanup/recovery 均 fail-close，card action 仍 durable unsupported
     - Task 5 最终 184 focused、277 affected、879 expanded、1590 full Autonomous 测试通过；
       EI-QUEUE-01、Ruff、文档、配置与 diff 门禁通过，独立 Spec/Code review 均批准且无
       Critical/Important
     - 已完成 Task 6：Router dispatch、attempt binding 与 dispatch committed 在同一锚定帧；
       fail-close，full projection replay 保持在短提交锁之外
     - 已完成 Task 6：每个 accepted attempt 最多一次调用现有 `_run_acp_session`；
       completed/failed/canceled/timeout/action_required 全部原子进入数据面，restart recovery
       不重跑 unknown dispatch，terminal/finalize CAS 冲突复用已暂存 Blob 并有界重试
     - 已完成 Task 6：员工进程环境、模型/profile/effort、persona、权限/capability 与 Context
       snapshot 均冻结进 immutable permit；Git/Shell 使用 shell-equivalent 默认拒绝策略，
       provider/secret 异常不泄露，真实子进程验证 Manager/Vault/peer secret 不继承
     - 已完成 Task 6：visible `/hire` 卡片到 typed request、Journal projection、
       `ProjectedAgentRegistry` 的模型三元组只组合一次；Hire admission 在 anchor 前拒绝
       composite/未知 effort/不支持 profile，legacy composite 语义保留为历史兼容，不再作为新入口。
     - Task 6 最终 224 affected、1686 full Autonomous 测试通过；changed-file Ruff、
       `git diff --check` 通过，独立 Spec/Code review 均批准且无 Critical/Important
     - 已完成 Task 7：`EmployeeDepartmentRuntime` 现拥有 Inbox/Router/Gateway/附件/data，
       unknown dispatch 收敛为 action_required，再启动有退避的 Journal 派发 worker；关停先停 ACK admission，
       超时未排空时保留 Journal/Vault 等依赖资源
       缺口稳定 fail-close；修复共享 Journal 推进后 Router workforce projection 过期导致合法消息
       `authority_stale` 的组合缺陷，并覆盖真实 anchored Inbox → owned Router queue
     - 已完成 Task 7：九个 `EI-*` exact selector 全部本地通过；全局 FI-29 只接受绑定
       `EI-IPC-01`、commit、构建 artifact 与结果摘要的严格 bridge，任意 `passed=true` 仍为 PENDING。
       本地 evidence 不能自我晋级为真实租户或外部 trust 证据；该阶段的 limit=0 发布门禁
       已被内置 single-host profile 取代，真实验收仍由部署方独立完成
     - Phase 3 最终 Autonomous `1700 passed, 2 skipped, 1 warning in 397.24s`
  4. Employee Response Channel：
     - **已完成**。实施与证据见 `docs/2026-07-14-autonomous-employee-response-plan.md`；采用稳定 UUID
       employee child create + 同一 child public `update_card` patch，禁止使用不能绑定预创建消息的
       非幂等 `channel.stream()` 路径
     - Journal-backed encrypted Durable Outbox、稳定 UUID 单卡、PREPARED/EXECUTING external Effects、
       monotonic/terminal fencing、restart retry 与 superseded snapshot GC 已完成
     - employee child create/update IPC 与 app/generation/connection/message receipt fence 已完成；
       Gateway terminal 卡只在原子执行终态提交后 append，runtime 负责恢复、worker、readiness 与逆序关闭
     - 员工 Bot 自己发送；主 Bot fallback 路径不存在。fresh Autonomous
       `1723 passed, 2 skipped, 1 warning in 397.51s`，共享回归 `193 passed`
  5. 团队 membership 与 `/stop` 终态竞态：
     - **已完成**。实施计划与官方 API 合约见
       `docs/2026-07-14-autonomous-membership-stop-plan.md`
     - canonical `employee_v1` membership 现由 Journal-backed 状态机管理；Manager Bot 使用
       `member_id_type=app_id` 串行调用官方群成员增删 API，目标员工 Bot 使用
       `members/is_in_chat` 证明最终状态。普通成员列表与 Chat API `chat_type` 均不再作为 Bot
       membership/会话结构依据
     - 团队 membership 管理在 handler 与 service 两层重验管理员/团队创建者、租户、ACTIVE visible
       当前 chat，不删除全局员工、credential 或其他群关系；legacy virtual role 保留独立旧路径
     - Bot added/deleted 事件已接入 low-level employee Channel，并先进入 Durable Inbox；事件只触发
       员工凭据 `is_in_chat` 对账，乱序/重复事件不会直接改投影。未知结果进入
       `DEGRADED/ACTION_REQUIRED`，Router 默认拒绝继续派发
     - `/stop` 在 Inbox ACK 后进入独立 durable control path，Router 入队前再次拦截；cancel request
       先锚定再撤销未执行 permit 或取消运行中 ACP。cancel 与 terminal 共享串行锁：terminal-first
       返回已结束，cancel-first 强制唯一 `canceled` 终态，迟到 ACP success 不可覆盖
     - `/stop` 权限为管理员、当前团队创建者或原任务发起人；命令结果通过员工 Durable Outbox
       投递，无主 Bot fallback。重启对已锚定 cancel 直接收敛 canceled，不重跑 ACP
       `269 passed`；`ruff check src/autonomous/`、配置校验与 `git diff --check` 通过
     - Phase 5 当时不提升旧 release readiness；后续内置员工决策已把默认 limit 调整为 8，
       显式设为 0 仍是关闭开关，下一阶段为 `/fire` durable Saga
  6. `/fire` durable Saga：
     - 已完成：`fire.requested`、`employee.state_changed=retiring` 与
       `employee.ingress.closed` 在同一 Journal frame 锚定，Router/Channel 新 ACK 立即 fail-close
     - 已完成：默认取消当前 attempt 并等待有界 grace；`--drain` 只等待自然完成且同样有上限。
       Slash 空集合经最终 GET 验证，Channel 停止、已知群 remove-only membership、Vault destroy、
       manifest fsync 与同文件系统原子 rename 依序执行
     - 已完成：六个外部/破坏性 Effect 均为 PREPARED → EXECUTING → COMMITTED；恢复时
       PREPARED 可继续，EXECUTING 只查询不重放。未知结果转 ACTION_REQUIRED，禁止继续销毁凭证或归档
     - 已完成：`archive_manifest.json` 包含文件 hash、history 日期范围、cleanup disposition、
       credential 销毁证明哈希及 `external_app_disposition=manual_deletion_required`；终态消息明确
       GhostAP 未删除开放平台应用，仍需管理员手动停用/删除
     - 已完成：同消息幂等重投、credential projection 崩溃补写、归档内容 hash 复核和 symlink 拒绝；
       `277 passed`，Ruff、配置校验和 `git diff --check` 通过
     - Phase 6 不提升 release readiness；visible limit 仍为 0，下一阶段是数据面真实
       producer/handler/Supervisor cutover
  7. 数据面真实 producer/handler/Supervisor cutover：
     - **已完成**。实施与审核见 `docs/2026-07-14-autonomous-data-cutover-plan.md`；canonical
       `execution_history.jsonl` 或根 `MEMORY.md`，legacy virtual role 行为保持不变
     - terminal history 与 Gateway/Router 终态继续原子锚定；completed attempt 在员工 Outbox terminal
       之前，以 attempt ID 幂等发布 L1、chat/thread `memory_summary`、skill profile 和 task-scoped reasoning。
       任一 canonical document sink 失败均阻止成功卡，restart 从加密 history 补齐文档而不重跑 ACP
     - 独立 `employee.data.authority_cutover` epoch/sequence 在 verified legacy import 后单向切至 canonical；
       canonical 写、read audit、preflight、Blob stage 与 document publish 均在落盘前重验 authority。
       malformed manifest、L1 冲突、多源 L1、symlink/non-regular source 或 live Blob 损坏全部阻断恢复
     - `/history` 与 ACL-aware `/memory` 使用 typed read ports。权限只由 transport principal/tenant/
       receiving app、官方消息 `chat_type` 和 workforce membership 推导；管理员宽读只允许主 Bot P2P，
       employee Channel `/memory` 只返回当前 chat/thread summary，授权和 durable audit 均早于 plaintext read
     - 主 Bot 新增独立 `/history <employee>` 与 `/employee-memory <employee>`，不混淆既有
       `/discuss history` 和 chat-scoped `/memory`；employee Channel 命令在 Inbox ACK 后、Router 前处理，
       结果只经员工 Durable Outbox 投递，无主 Bot fallback
     - runtime recovery 验证全部 live history/document Blob，重建 daily JSONL 和 L1/summary/skill/reasoning
       materialization；history 分片和查询统一使用配置时区。多轮 grill 自动采纳并关闭伪 membership、
       data writer 漂移、summary terminal 崩溃窗口、source_id 链、root MEMORY 冲突和 migration symlink
     - fresh Autonomous `1791 passed, 2 skipped, 1 warning in 401.32s`；受影响回归 `265 passed`，
     - Phase 7 完成后，Phase 8 将员工能力从外部发布门禁切换为内置本地启动
  8. 内置 Visible Employee 启动：
     - 默认 visible limit 为 8；显式 0 是运维关闭开关
     - 首次启动原子创建 versioned Journal/Vault/Data 三把 256-bit key，目录 0700、文件 0600；
       symlink、错误 owner/mode、损坏或显式/自动密钥混用均 fail-close
     - `EmployeeDepartmentRuntime` 直接使用本地 FileAnchor、恢复 Journal 并注入 `/hire` service；
       release broker、evidence、lease、人工 sandbox attestation 不再参与启动或 readiness
     - 主 Bot 本地审计仅作诊断；员工创建和激活不依赖外部 ledger
     - Channel 优先 bwrap，失败后清理旧进程与 pipe，再以 `process-fallback` 重试一次；fallback 永远
       保持 `verified=false`，凭据仍只通过一次性 pipe 下发
     - 飞书 `aregister_app()` 注册链接仍由发起用户打开并按官方指引完成创建，这是唯一人工步骤，
       不要求租户管理员审批
     - 一键创建改用官方智能体 preset，并显式包含 Bot-to-Bot mention、云文档评论、Slash Command、
       卡片及 WebSocket 事件所需权限；生产 low-level Channel 已使用员工自身 `lark-oapi` client 完成
       text/card/post、reply、card patch 和 comment reply 出站，`/status` 激活不再命中
       `outbound-transport-separate`
     - 权限/事件/出站 primitive 不等于工作流闭环：自动 Bot-to-Bot 接力仍需同群 ACTIVE membership、
       Bot sender authority 与循环预算；云文档评论执行仍需 comment event normalization、逐文档授权、
       comment fetch 与 durable route，未完成前不得标为可用
  9. 真飞书租户体验与验收：
     - 2026-07-17 自动化前置实现已完成：持久员工 workspace/bootstrap/Actor、动态 Team
       Coordinator、群路由、恢复、知识 Wiki、状态卡和管理员恢复动作均已有本地合同
     - `shadow` 模式现只执行一次 legacy 模型调用，同时锚定 Actor 输入与 workspace/context
       digest 对比；不一致只写 secret-free audit，不改变用户结果
     - 真实租户 manifest 新增 persistent actor、direct mention、Team review/revision、partial
       context、selective wake 与 Fire 六组门禁；1/10/50 soak 继续强制
     - 2026-07-17 默认已切到 `actor` / `coordinator`；旧 settings 对象缺字段时生产组合根也
       选择持久模式，Actor/Coordinator 失败不会自动回退
     - `legacy_one_shot` / `legacy_pipeline` 仅保留为显式紧急回滚；固定流水线和 one-shot
       生产路径要等真实租户签名证据通过后删除
     - 验收 CLI 可生成绑定 release/commit/service/tenant 的 fail-closed capture 模板，所有断言
       初始为 false，不能被当作真实租户通过证据
     - staging + production Provisioning
     - 真实员工独立收发、桌面/移动 Slash、附件/话题/CardAction
     - 1/10/50 Bot soak、断线/重启/限流/故障注入
     - 验收失败形成缺陷并修复，但不再通过外部发布系统控制本地启动

  每个阶段完成后都必须检查原始附件中的要求，不能把子计划完成当作整体完成。

  最终 completion audit 必须逐项证明：

  - `/hire` 真实创建独立飞书应用并得到 Vault credential ref。
  - 员工 Channel 能真实收消息、断线恢复。
  - Slash Commands 服务端 desired set 精确一致。
  - 多员工消息正确路由，员工 Bot 自己回复。
  - `_run_acp_session` 每 accepted attempt 只调用一次。
  - completed/failed/canceled/timeout/action_required 都有历史。
  - history 日期范围、ACL、重建正确。
  - Thread 全量 Context 与 watermark/failure 正确。
  - `/stop` 终态竞态正确。
  - `/fire` 可恢复且不伪报删除开放平台应用。
  - 主 Bot send count 在员工响应中为 0。
  - Deep/Spec/Workflow 和主 Bot WS 无回归。
  - 自动化、故障注入、重启恢复、配置校验全部通过。
  - 真实飞书租户 E2E 完成。
  - dev 已 push，HEAD 与 origin/dev 一致。

  只有所有要求都有当前、权威证据时，才能宣布整体目标完成。
  如果只完成了某个阶段，要明确说明整体目标仍 active，并继续下一个阶段。
