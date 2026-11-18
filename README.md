# GhostAP

GhostAP 是一个面向受信工程环境的飞书/Lark 原生 AI 研发控制面与执行网关。主 Bot 作为统一控制面管理项目、受管群、直接编程会话及长任务引擎；底层 provider/transport、模型和执行引擎均可替换，普通编程仍可直接连接用户选定的 Agent。

完整边界以 [产品合同](docs/product-contract.md) 为准。

## 核心能力

| 能力 | 产品边界 |
| --- | --- |
| 主 Bot 控制面 | 管理项目、受管群、任务调度、锁控制、权限引导和投递审计 |
| 直接编程 | Coco、Claude、Aiden、Codex、Gemini、Traex、Grok、DSH、AGY 后端保持多轮直连 |
| 长任务引擎 | Deep 单次规划自主执行、Spec 结构化闭环审查、Workflow 多 Agent JS 编排 |
| 飞书交互 | 卡片持续展示任务状态、工具调用、模型选择和错误诊断 |
| Host Shell | 直接宿主机执行；GhostAP 不增加命令过滤或操作系统沙箱 |
| 本地持久化 | 项目状态与受管群注册表面向单机文件存储，不承诺多副本线性一致性或对特权宿主机的回滚抵抗 |

## 运行模型

GhostAP 把产品身份、执行策略和 provider/transport 拆开：

| 维度 | 说明 |
| --- | --- |
| 产品身份 | 主 Bot 为统一控制面与消息交互入口 |
| provider/transport | ACP 直接模式、Shell CLI 桥接 |
| Host Shell | 独立的直接宿主执行能力，不是 Agent provider |

普通工具入口会设置聊天 + 项目的持续模式，直到 `/exit`。Deep、Spec 和 Workflow 是作用在话题/根线程上的任务引擎，不会替换普通编程模式。普通编程、Deep、Spec 和 Workflow 收到任务后均会自动推进到成功或明确失败；`/wf` 自动绑定可用推荐 Agent Pool 并开始编排，不要求手动确认或批准继续。Smart 是默认模式；当 `DEFAULT_ACP_TOOL` 留空时，未匹配的自由文本会按 Shell 命令处理。

## 快速开始

### 环境要求

- Python 3.11+
- `uv`
- 飞书/Lark 企业自建应用，开启长连接接收事件
- 如需使用客户端 Slash Command 面板，飞书 PC 端需 7.70+，移动端需 7.71+
- 如需使用 `/wf`，需要 Node.js 20+
- 需要使用的 AI 工具或 ACP Provider 已在本机安装并完成各自认证

### 安装依赖

```bash
uv sync --group dev
```

也可以直接执行 `./restart.sh start`。该入口默认先运行 `uv sync --group dev`；
`uv` 本身仍是启动前唯一需要预装的 Python 包管理工具。可用
`GHOSTAP_SYNC_PYTHON_DEPENDENCIES=0` 跳过依赖同步。

### 配置飞书应用

在飞书开放平台创建企业自建应用后，至少配置：

1. 获取 `APP_ID` 和 `APP_SECRET`。
2. 在“事件与回调”中启用“使用长连接接收事件”。
3. 订阅 `im.message.receive_v1`。
4. 在“回调订阅”中只保留新版 `card.action.trigger`（卡片回传交互）；
   删除旧版 `card.action.trigger_v1` 或旧请求地址配置。GhostAP 不再接收旧回调，
   也不再发送 Card JSON 1.0。回调配置以[飞书官方最新版说明](https://open.larkoffice.com/document/feishu-cards/card-callback-communication)为准。
5. 授权消息接收、消息发送和卡片更新相关权限。
6. 授权 `application:app_slash_command:read` 和
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
INGRESS_ACCESS_MODE=legacy_allow_all
ADMIN_BOOTSTRAP_SCOPE=p2p_only
```

常用配置：

```env
COMMAND_TIMEOUT=30
COMMAND_MAX_OUTPUT_LENGTH=4000
ACP_MODEL_PROBE_TIMEOUT=15

WORKFLOW_TOTAL_TIMEOUT_S=3600
WORKFLOW_AGENT_CALL_TIMEOUT_S=600
WORKFLOW_SCRIPT_GEN_TIMEOUT_S=180

```

`INGRESS_ACCESS_MODE` 默认使用 `legacy_allow_all`，当前飞书 App 收到的私聊和群聊可
直接使用 `/dsh` 等普通功能；需要用户与群双白名单隔离时可显式改为 `enforced`。
该配置和管理员列表只控制谁可以通过飞书入口操作服务；Shell、
ACP 与 Workflow 执行不再叠加 GhostAP 自己的风险策略、工具过滤或
进程沙箱。Codex、Claude 等后端按各自权限模型处理执行请求。

更多参数见 `.env.example` 和 `src/config/settings.py`。各 AI 后端所需的密钥、登录态或 CLI 配置应按对应工具自己的方式准备，GhostAP 只读取必要的环境变量和本地命令。

### 校验并启动

```bash
uv run python -m src.main --validate
uv run python -m src.main
```

`--validate` 检查必需配置和运行依赖，发现阻断项时返回非零。

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
| `/coco`、`/claude`、`/aiden`、`/codex`、`/gemini`、`/traex`、`/grok`、`/dsh`、`/agy` | 进入对应编程模式 |
| `/<工具> <需求或原生命令>` | 单句直达对应工具执行（如 `/agy /swarm 修复测试`、`/codex /review`） |
| `/cmds [工具]` | 打开工具原生 `/` 命令与 Skills 卡片（并支持一键执行无参命令） |
| `/coco_info`、`/claude_info`、`/aiden_info`、`/codex_info`、`/gemini_info`、`/traex_info`、`/grok_info`、`/dsh_info`、`/agy_info`、`/info` | 查看当前或指定编程会话、模型与上下文用量 |
| `/model`、`/model list`、`/model <name>` | 自动适配查看或热切换当前工具模型 |
| `/acp [工具]` | 查看或热切换项目默认 ACP 工具 |
| `/stop`、`/clear` | 立即停止运行中任务（不退出模式）/ 清空上下文并保留工具与模型 |
| `/sh <命令>`、`/ask <工具> <问题>` | 任意模式执行 Shell / 一次性零副作用旁路咨询 |
| `/exit` | 退出当前模式，回到 Smart |

工具原生 `/` 命令（如 AGY 的 `/swarm`、`/boost`、`/plan`、`/compact`，Codex/Traex/Coco 的 `/review`、`/review-branch`、`/mcp`、`/skills`）在对应编程模式下可直接发送透传；在 Smart 模式下发送专属原生命令（如 `/swarm ...`）也会自动路由到相应工具执行（`/logout` 等高危账号命令默认拦截）。

Host Shell 不需要单独入口；在 Smart 模式中，匹配为 Shell 的文本（或任意模式下使用 `/sh <命令>`）会进入宿主机
执行路径。命令会直接交给宿主 shell；GhostAP 仅保留超时、输出截断和日志脱敏，
不再做黑白名单、风险分级或操作系统隔离。

### 项目

| 命令 | 作用 |
| --- | --- |
| `/projects` | 查看项目面板 |
| `/new <名称> [目录]` | 创建项目 |
| `/switch <名称>` | 切换项目 |
| `/close <名称>` | 关闭项目 |
| `/status`、`/status <task_id>` | 打开工作区驾驶舱（工作区/Git分支/Direct/Workflow·Deep·Spec/锁）或查看任务详情 |

### 长任务引擎

| 命令 | 作用 |
| --- | --- |
| `/deep <需求>` | 单次规划并自主执行 |
| `/deep_status`、`/stop_deep` | 查看或停止 Deep |
| `/spec <需求>` | 按 Spec → Plan → Task → Build → Review 闭环推进 |
| `/spec_status`、`/stop_spec` | 查看或停止 Spec |
| `/wf <需求>` | 自动绑定推荐 Agent Pool，随后生成、验证并执行 JS Workflow |
| `/wf_status`、`/wf_help`、`/stop_wf` | 查看进度、帮助或停止 Workflow |
| `/workflow`、`/workflow_status`、`/workflow_help`、`/stop_workflow` | Workflow 长命令别名 |

Spec 同时支持编程、写作、研究和其他领域任务。规格阶段先从目标、受众、交付物和风险推断所需专家，再按各角色的关注点和必查项进行深度审查；写作覆盖思路、读者、编辑与审稿，并按题材补充领域专家。完成度继续独立验收，自动迭代、恢复和停止规则保持一致。

Spec 的阶段、任务进度、子代理输出和多角色审查结论汇总到主卡，不再按子任务或迭代轮次发送新卡。只有内容达到卡片容量边界时续接，历史内容保留。

## 全自动执行

普通编程、Deep、Spec 和 Workflow 使用已保存或自动绑定配置，缺失时采用可用的推荐工具和默认模型，经过有界自动恢复后到达成功或明确失败终态。`/wf <需求>` 自动绑定包含 1–8 个 `tool+model` Agent 的推荐 Agent Pool（`A1`、`A2` 等稳定标识）与自动选择的主编排器；池绑定后即冻结，脚本生成、验证与执行全自动推进，不要求确认或继续批准门。

普通选择自动采用推荐项。GhostAP 不再对命令或 ACP 工具做二次风险判定；执行权限由 Codex、Claude 等后端自身负责。

Workflow 动态生成任务专用 JS，并按复杂度组合 `agent()`、`sequence()`、`fanout()`、`verify()`、`generate()`、`tournament()`、`loop()` 和 `race()`。脚本中的直接和动态调用只能引用已确认池内的 Agent，不能覆盖冻结的工具和模型绑定。脚本运行在普通 Node.js 宿主上下文，可使用 Node 模块、进程环境和网络；运行时只负责确定性控制流与调用预算。进度卡片展示任务、完整池、静态计划、动态候选和直接子 Agent 的实际调度状态，每个 Agent 只保留一条最新操作，完整结果通过分页或附件交付。

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

**关键依赖：**
- `lark-oapi==1.7.2`：REST API 消息发送、卡片更新、开放平台交互
- `lark-channel-sdk==1.2.0`：WebSocket 事件订阅与接收

## 访问与运维

- Host Shell、ACP 和 Workflow 都直接使用宿主机能力；GhostAP 不提供
  命令风险策略、工具过滤、Bubblewrap、Seatbelt 或 Node VM 隔离。
- 飞书消息有过期检查和去重缓存，避免重复执行。
- ACP 权限请求采用后端提供的授权选项，由后端权限模型负责。
- 仓库操作受 repo 锁保护，群聊访问可由管理员锁定。
- 卡片按钮带签名校验，错误详情会脱敏和截断。
- Workflow 只验证模块结构、Agent Pool 绑定和调用预算，不限制 Node 模块。
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
│   ├── card/            # 飞书卡片事件、渲染、投递和状态管线
│   │   ├── actions/
│   │   ├── delivery/
│   │   ├── events/
│   │   ├── render/
│   │   ├── session/
│   │   ├── state/
│   │   ├── timers/
├── tests/               # 测试
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
