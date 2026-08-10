# GhostAP

GhostAP 是一个面向受信工程环境的飞书/Lark 原生 Agent Department
Gateway。主 Bot 是控制面，负责项目、员工、团队、任务、审批和审计；员工 Bot
是拥有独立 Channel、历史、记忆和停止语义的执行身份。底层
provider/transport、模型和执行引擎均可替换，普通编程仍可直接连接用户选定的
Agent。

完整边界以 [产品合同](docs/product-contract.md) 为准。

## 核心能力

| 能力 | 产品边界 |
| --- | --- |
| 主 Bot 控制面 | 管理项目、员工、团队、任务、审批和审计，不冒充员工输出 |
| 直接编程 | Coco、Claude、Aiden、Codex、Gemini、Traex 后端保持多轮直连 |
| Agent Department | 持久员工拥有独立飞书 Bot、Channel、历史、记忆和停止语义 |
| 飞书交互 | 卡片持续展示任务状态、工具调用、模型选择和错误诊断 |
| Host Shell | 特权宿主机执行，只能关闭或显式授权；超时、截断和命令过滤不构成操作系统沙箱 |
| 本地持久化 | Journal、Vault、Blob 和项目状态面向单机文件存储，不承诺多副本线性一致性或对特权宿主机的回滚抵抗 |

## 运行模型

GhostAP 把产品身份、执行策略和 provider/transport 拆开：

| 维度 | 说明 |
| --- | --- |
| 产品身份 | 主 Bot 是控制面；Employee Bot 是独立执行身份 |
| provider/transport | ACP 直接模式、Shell CLI 桥接 |
| Host Shell | 独立的特权宿主执行能力，不是 Agent provider，也不是操作系统沙箱 |

普通工具入口会设置聊天 + 项目的持续模式，直到 `/exit`。Deep、Spec 和 Workflow 是作用在话题/根线程上的任务引擎，不会替换普通编程模式。四种执行策略收到任务后都会自动推进到成功或明确失败，不以 Agent 选择、Review、脚本确认或手动恢复阻塞主路径。Smart 是默认模式；当 `DEFAULT_ACP_TOOL` 留空时，未匹配的自由文本会按 Shell 命令处理。

## 快速开始

### 环境要求

- Python 3.11+
- `uv`
- 飞书/Lark 企业自建应用，开启长连接接收事件
- 如需使用客户端 Slash Command 面板，飞书 PC 端需 7.70+，移动端需 7.71+
- 如需使用 `/wf`，需要 Node.js 20+
- 需要使用的 AI 工具或 ACP Provider 已在本机安装并完成各自认证
- 可见员工 Channel 在 Linux 使用 `bubblewrap`，在 macOS 使用系统 Seatbelt；
  `restart.sh` 会自动同步 Python 依赖、安装受支持 Linux 发行版的
  `bubblewrap`，或探测 macOS 自带的 `/usr/bin/sandbox-exec`

### 安装依赖

```bash
uv sync --group dev
```

也可以直接执行 `./restart.sh start`。该入口默认先运行 `uv sync --group dev`
并准备当前平台的员工隔离依赖；`uv` 本身仍是启动前唯一需要预装的 Python
包管理工具。可分别用 `GHOSTAP_SYNC_PYTHON_DEPENDENCIES=0` 和
`GHOSTAP_PREPARE_EMPLOYEE_SANDBOX=0` 跳过这两步。

### 配置飞书应用

在飞书开放平台创建企业自建应用后，至少配置：

1. 获取 `APP_ID` 和 `APP_SECRET`。
2. 在“事件与回调”中启用“使用长连接接收事件”。
3. 订阅 `im.message.receive_v1`。
4. 授权消息接收、消息发送和卡片更新相关权限。
5. 授权 `application:app_slash_command:read` 和
   `application:app_slash_command:write`，然后创建并发布新的应用版本。

服务启动后会在后台通过官方 OpenAPI 对账主 Bot 的 Slash Command 面板，
注册 GhostAP 当前支持的主要命令；Channel SDK 仍负责接收用户选择命令后产生的
普通消息事件。飞书单个应用最多支持 100 条 Slash Command，GhostAP 只展示主要
拼写，`/workflow`、`/enter_codex` 等兼容别名仍可直接发送。

Slash Command 创建或更新后通常约 5 分钟生效，客户端还可能缓存约 3 分钟；
若面板暂未刷新，可等待后重启飞书客户端。缺少上述权限时主 Bot 不会停止服务，
日志会记录可操作的同步告警，补权限并发布应用版本后重启 GhostAP 即可重新对账。

### 配置环境变量

```bash
cp .env.example .env
vim .env
```

最小配置：

```env
APP_ID=your_app_id
APP_SECRET=your_app_secret
DEFAULT_ACP_TOOL=coco
ADMIN_USER_IDS=
INGRESS_ACCESS_MODE=enforced
ADMIN_BOOTSTRAP_SCOPE=p2p_only
SHELL_ACCESS_MODE=disabled
EMPLOYEE_DEPARTMENT_ENABLED=false
EMPLOYEE_GROUP_CONTEXT_RETENTION_DAYS=30
```

常用配置：

```env
SANDBOX_TIMEOUT=30
SANDBOX_MAX_OUTPUT_LENGTH=4000
SANDBOX_COMMAND_BLACKLIST=

ACP_PERMISSION_AUTO_APPROVE=true
ACP_MODEL_PROBE_TIMEOUT=15

WORKFLOW_TOTAL_TIMEOUT_S=3600
WORKFLOW_AGENT_CALL_TIMEOUT_S=600
WORKFLOW_SCRIPT_GEN_TIMEOUT_S=180

```

这些字段构成显式安全姿态：空授权列表不代表公开访问；Host Shell 默认关闭；
Employee Department 必须单独启用，并为群上下文设置有界保留期。`shadow`、
`legacy_allow_all` 和 `trusted_local` 只用于显式诊断或紧急回退，校验输出会用稳定
finding code 标出未强制执行或未确认的风险。

更多参数见 `.env.example` 和 `src/config/settings.py`。各 AI 后端所需的密钥、登录态或 CLI 配置应按对应工具自己的方式准备，GhostAP 只读取必要的环境变量和本地命令。

### 校验并启动

```bash
uv run python -m src.main --validate
uv run python -m src.main
```

`--validate` 会输出 `[安全姿态]`；存在 blocking finding 时返回非零，必须先修正
配置或准备隔离后端。

首次启动后，可在飞书私聊机器人发送 `/setadmin` 设置管理员。`ADMIN_USER_IDS` 为空时允许首次设置；设置后只有管理员可以替换管理员配置。

### 安全远程重启

从 Bot 任务中需要重启当前 checkout 时使用：

```bash
./restart.sh rr
```

`rr` 会在同步预检时固定当前服务 generation；预检成功后，跨进程门禁先阻止新
任务进入，再等待正在运行的任务完成终态记账，最后才执行停止、启动和 readiness
检查。同一 generation 上的并发请求只会执行一次重启。默认 7200 秒预算从独立
worker 启动后计算，覆盖延迟、任务排空、重启操作和超时进程组清理；超时以非零
状态退出，不会留下失控的重启子进程。同步预检本身不计入该预算，以便配置、门禁
和 generation 错误直接返回给调用方。新进程只有在飞书 WebSocket 已连接、精确
PID 与启动指纹均匹配后才发布 ready generation；仅存活但未就绪的进程会被清理，
不会被报告为启动成功。停止阶段默认给主进程 30 秒优雅退出时间，之后才会强制
终止进程组；发生强制终止时会明确返回降级的非零状态。

门禁目录默认位于 checkout 外的同级
`.ghostap-restart-gates/<checkout-hash>/gate`，因此清理或替换 checkout 不会
替换仍被运行任务持有的 lock inode。如用
`GHOSTAP_RESTART_GATE_DIR` 指定覆盖目录，它必须是私有、专用的绝对目录，且既有
目录须归当前用户所有并禁止 group/other 访问；`/`、`/tmp`、用户主目录或 checkout
根目录等宽泛路径会被拒绝。首次绑定后 locator 会固定门禁身份，运行中修改覆盖
目录会 fail-closed。`GHOSTAP_RESTART_GATE_TIMEOUT` 可调整共享预算秒数。

## 常用命令

### 模式与模型

| 命令 | 作用 |
| --- | --- |
| `/help` | 查看完整帮助 |
| `/coco`、`/claude`、`/aiden`、`/codex`、`/gemini`、`/traex` | 进入对应编程模式 |
| `/model`、`/model list`、`/model <name>` | 查看或切换当前 ACP 工具模型 |
| `/acp` | 查看 ACP 工具选择入口 |
| `/exit` | 退出当前模式，回到 Smart |

Host Shell 不需要单独入口；在 Smart 模式中，匹配为 Shell 的文本会进入宿主机
执行路径。它是特权能力而非操作系统沙箱，按产品合同必须关闭或由授权用户显式
启用；黑白名单、超时和输出截断只是附加防护。

### 项目

| 命令 | 作用 |
| --- | --- |
| `/projects` | 查看项目面板 |
| `/new <名称> [目录]` | 创建项目 |
| `/switch <名称>` | 切换项目 |
| `/close <名称>` | 关闭项目 |
| `/status` | 查看当前项目、模式、锁和任务状态 |

### 长任务引擎

| 命令 | 作用 |
| --- | --- |
| `/deep <需求>` | 单次规划并自主执行 |
| `/deep_status`、`/stop_deep` | 查看或停止 Deep |
| `/spec <需求>` | 按 Spec → Plan → Task → Build → Review 闭环推进 |
| `/spec_status`、`/stop_spec` | 查看或停止 Spec |
| `/wf <需求>` | 生成并执行 JS Workflow 编排脚本 |
| `/wf`、`/wf_status`、`/wf_help`、`/stop_wf` | 生成、查看或停止 Workflow |

### Agent Department（持久数字员工）

| 命令 | 作用 |
| --- | --- |
| `/hire <名字>` | 由配置管理员在主 Bot 私聊中雇佣持久数字员工 |
| `/hire <名字> --tool codex --model <模型> --role coder` | 使用受控参数发起雇佣 |
| `/employees` | 查看在职数字员工 |
| `/fire <名字>` | 退役持久数字员工 |
| `/history <名字>`、`/employee-memory <名字>` | 由主 Bot 管理员读取授权范围内的员工历史或记忆 |

**雇佣数字员工流程（/hire）：**

1. 配置管理员在主 Bot 私聊发送 `/hire 小明`，打开工具和模型选择卡片。
2. 选择后启动 Journal-backed 雇佣流程，并按返回的飞书注册链接完成应用创建。
3. 凭据写入加密 Vault，独立 Channel、Slash Commands 和身份校验全部就绪后，
   员工才进入可用状态；等待或失败不会被报告成创建成功。

可选参数直接写在命令行跳过卡片交互：

```
/hire 小明 --tool codex --model <模型名> --role coder --profile standard --effort default
```

员工创建后：

- Journal、加密 Blob/Vault 是事实源，`identity.json` 仅是可安全重建的投影，不含密钥。
- 员工使用自己的 Bot 接收任务、更新卡片和返回结果，不回退到主 Bot 代发。
  `/memory` 和 `/stop` 管理其工作。
- `/hire` 拒绝任意提示词注入；工作风格由受控 role/profile 与持久上下文形成。

旧的独立 Autonomous Manager 命令面已经退役并默认拒绝，不是 Agent Department
的生产入口。

## 全自动执行

普通编程、Deep、Spec 和 Workflow 共用一条主路径契约：使用已保存配置，缺失时采用可用的推荐工具和默认模型，经过有界自动恢复后到达成功或明确失败终态。用户只需在运行中主动停止，或通过独立配置入口改变默认值。

普通、安全、可逆的选择自动采用推荐项；高风险且未经原始请求精确授权的操作拒绝或跳过，并继续可安全完成的部分。ACP 权限保持 fail-closed，不通过跳过权限检查换取自动化。

Workflow 动态生成任务专用 JS，并按复杂度组合 `agent()`、`sequence()`、`fanout()`、`verify()`、`generate()`、`tournament()`、`loop()` 和 `race()`。运行时负责确定性控制流、总 Agent 数和危险能力限制，Agent 负责语义工作；简单任务保持单 Agent。进度卡片展示任务、阶段和直接子 Agent 的调度状态，每个 Agent 只保留一条最新操作，完整结果通过分页或附件交付。

## 架构入口

| 路径 | 说明 |
| --- | --- |
| `src/main.py` | 应用启动、配置校验和生命周期 |
| `src/feishu/ws_client.py` | 飞书 WebSocket 入口、消息校验、去重和调度 |
| `src/feishu/handlers/` | 命令处理器 |
| `src/mode/` | 聊天/项目交互模式状态 |
| `src/acp/` | ACP 会话、Provider、模型发现、诊断和事件渲染 |
| `src/agent_session/` | ACP 与 CLI 后端的统一会话抽象 |
| `src/deep_engine/` | Deep 单次自主执行 |
| `src/spec_engine/` | Spec 结构化闭环和多视角审查 |
| `src/workflow_engine/` | JS Workflow 生成、验证、运行时和卡片渲染 |
| `src/autonomous/` | v5 自主工作系统（详见下方） |
| `src/card/` | CardSession 事件管线、纯渲染和卡片投递 |
| `src/project/`、`src/project_chat/`、`src/thread/` | 项目、群绑定和线程上下文 |
| `src/chat_lock.py`、`src/repo_lock.py`、`src/utils/lock_order.py` | 聊天锁、仓库锁和锁顺序约束 |
| `src/config/` | Pydantic Settings 和 `.env` 配置 |

卡片管线遵循单向依赖：

```text
handler -> session -> render
                  -> delivery
```

渲染层保持纯函数；投递层不反向依赖会话层。跨层共享类型放在 `src/card/protocols.py` 或 `src/card/events/`。

## Agent Department 耐久架构（src/autonomous/）

生产 Employee Department 使用 Journal-backed 持久化架构，所有状态变更通过事务帧记录，并通过 Vault、Blob、独立员工 Channel 和 Durable Outbox 完成恢复。生产组装根位于 `provisioning/composition.py`，现役执行面由 `gateway/coordinator.py`、`runtime/employee_actor.py` 和 `supervisor/employee_channels.py` 构成；旧 Goal/Run Manager 不再是产品入口。

**关键依赖：**
- `lark-oapi==1.7.1`：REST API 消息发送、卡片更新、机器人管理
- `lark-channel-sdk==1.1.0`：WebSocket 事件订阅（持久收件箱）

**测试：**

```bash
uv run python -m pytest tests/autonomous/ -q
uv run ruff check src/autonomous/         # 0 错误
```

## 安全与运维

- Host Shell 是特权宿主机执行，不是操作系统沙箱；仅限受信工程主机，并且必须
  保持关闭或由授权用户显式开启。命令过滤、超时和输出截断不提升隔离等级。
- 飞书消息有过期检查和去重缓存，避免重复执行。
- ACP 工具调用通过权限钩子处理，可配置自动批准或默认拒绝。
- 仓库操作受 repo 锁保护，群聊访问可由管理员锁定。
- 卡片按钮带签名校验，错误详情会脱敏和截断。
- Workflow 脚本会做结构化验证，禁止危险模块和明显逃逸。
- 可见员工 Channel 在 Linux 通过 Bubblewrap user/mount/PID namespace 验真；
  macOS 通过 deny-default Seatbelt 与凭证下发前拒绝探针验真。macOS 缺少
  系统 Seatbelt 时员工 Channel 默认拒绝启动，不会伪报隔离成功。
- macOS 的 `sandbox-exec` 是系统兼容接口且已被 Apple 标记为 deprecated；
  当前实现需要在目标 macOS 版本上完成 profile、DNS/TLS/WSS 真机验收。需要
  Apple 长期支持的产品边界时，应迁移到签名 helper + App Sandbox entitlement。
- 日志优先查看 `logs.log`；重启或启动问题同时检查 `[RESTART]` 标记和 `uv run python -m src.main --validate` 输出。

## 开发

本仓库只使用 `uv`：

```bash
uv sync --group dev
uv run python -m src.main --validate
uv run python -m pytest tests/ -q
uv run python -m pytest tests/test_acp_client.py -q
uv run ruff check .
```

针对性修改时先跑最相关测试；涉及共享路由、卡片渲染、锁、配置或会话代码时扩大测试范围。项目约定见 `AGENTS.md`，提交信息规范见 `docs/commit-message-guidelines.md`。

## 目录

```text
ghostAp/
├── src/                 # 应用代码
│   ├── autonomous/      # v5 自主工作系统
│   ├── card/            # 飞书卡片事件、渲染、投递和状态管线
│   │   ├── actions/
│   │   ├── delivery/
│   │   ├── events/
│   │   ├── render/
│   │   ├── session/
│   │   ├── state/
│   │   ├── timers/
├── tests/               # 测试
│   ├── autonomous/      # 自主系统测试（unit/integration/chaos/security/contract）
├── docs/                # 架构记录和接入指南
├── scripts/             # 辅助脚本
├── ux/                  # UI 预览和验证资产
├── .Memory/             # 近期决策、验证和风险记录
├── AGENTS.md            # AI 编码代理项目指令
├── .env.example         # 环境变量模板
├── pyproject.toml       # Python 项目配置
└── README.md
```

## License

MIT License
