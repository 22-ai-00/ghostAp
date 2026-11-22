# AGY ACP 与 Orchestrator CLI 混合接入

## 运行拓扑

GhostAP 将 Google Antigravity CLI（`agy`）作为一等编程后端，按所选模型配置自动路由到最合适的官方传输通道：

1. **常规模型与官方默认 Orchestrator 预设（原生 ACP stdio）**：
   ```text
   GhostAP -> agy acp (stdio JSON-RPC) -> Antigravity Agent
   ```
   适用于所有单个模型（含 `low`/`medium`/`high`/`xhigh` 原生 `thought_level`）以及默认预设的 `gemini-pro-orchestrator*`（`Planner Effort = default`），也保持完整 Native Swarm 生命周期、原生 `/` 命令、`request_permission` 与 `is_cli_backend("agy") == False`。
2. **Orchestrator 自定义 Planner/Executor Effort（CLI `stream-json`）**：
   ```text
   GhostAP -> agy --prompt=... --output-format stream-json --dangerously-skip-permissions --model <orch_id> --settings <json> [--conversation <id>]
   ```
   `agy acp`（1.5.14）尚未对 `gemini-pro-orchestrator*` 暴露 `thought_level` 或按会话隔离的 `orchestrator` 配置项；因此在选择自定义 `<orch_id>/<planner_effort>/<executor_effort>` 时，GhostAP 通过每进程独立 `--settings '{"showThinking": true, "orchestrator": {"planner": "<Planner Display Label>", "executor": "<Executor Display Label>"}}'` 启动并跨轮以 `--conversation` 续接会话，不篡改宿主 `~/.gemini/antigravity-cli/settings.json`。

可用性探测仍通过 `agy acp --help` 检查 `usage:`、`agy acp` 与 `stdio` 关键字，确保在服务就绪与工具面板阶段即可验证当前主机安装的 `agy` CLI 支持 ACP stdio server。

## 用户入口

- `/agy` 或 `/enter_agy`：进入持久 AGY 编程模式。
- `/agy_info`：查看当前 AGY 会话与模型/Orchestrator Effort 信息。
- `/model`：共享模型命令，自动适配当前 Codex、Traex、DSH、AGY 等编程后端；
  - 在 AGY 模式中实时探测 `model`、`thought_level`（或 `effort`）与 `availableModels` 展示标签，为常规模型生成模型/Effort 级联，为消费级账号的 `gemini-pro-orchestrator*` 生成 **Planner Effort**（`default` + Planner 模型支持的档位）与 **Executor Effort**（Executor 模型支持的档位）三级级联。
  - `/model <slug>` 同时兼容 AGY CLI 连字符别名（如 `gemini-4-neon-xhigh` → `gemini-4-neon/xhigh`）与 Orchestrator 两段简写（如 `gemini-pro-orchestrator-neon/xhigh`）。
- `/effort`：共享 Effort 查看与切换命令（例如 `/effort xhigh`、`/effort xhigh low`、`/effort xhigh/low`、`/effort default`）。
- `/exit`：共享退出命令，退出当前 AGY 或其他编程模式。
- Deep、Spec 与 Workflow Agent Pool 均可使用 `agy`；跨原生 ACP 与 CLI 自定义 Orchestrator 切换模型/Effort 时，GhostAP 会自动原子替换底层会话传输，同一传输内则直接热更新。

## 协议细节

### 原生 ACP（常规模型与默认 Orchestrator）

`agy acp` 不通过 CLI 参数传入 `--model`，而是始终以无模型参数方式启动 `agy acp`，并在 `session/new` 之后通过 ACP `session/set_config_option` 应用模型选择：

- 模型：`configId="model"`，选项来自实时 `session/new` 返回的 `configOptions`（如 `gemini-4-neon`、`gemini-4-carbon`、`gemini-4-argon`、`gemini-4-hydrogen`、`gemini-pro-orchestrator-neon`、`gemini-pro-orchestrator`、`gemini-pro-orchestrator-argon`、`claude-opus-5-5`、`gpt-oss-120b-medium` 等）。
- 思考档位：`configId="thought_level"`（兼容 `reasoning_effort`/`effort`），由 GhostAP 在模型发现阶段逐模型切换 `model` 并读取对应的实时档位选项（例如 `gemini-4-neon` / `gemini-4-hydrogen` 支持 `low`/`medium`/`high`/`xhigh`，`gemini-4-carbon` / `claude-opus-5-5` 支持 `low`/`medium`/`high`，而 `gpt-oss-120b-medium` 不暴露思考档位）。

### Orchestrator Planner/Executor Effort（消费级实时合成 + CLI `stream-json`）

在非 Enterprise 档位下，如果当前 `agy acp` 未对 Orchestrator 原生暴露 `thought_level`，且其底层 Planner 与 Executor 基础模型均在本次实时探测中返回有效档位，GhostAP 会为该 Orchestrator 合成 `default`（保持原生 ACP 预设）及 `<orch_id>/<planner_effort>/<executor_effort>` 组合：

- `gemini-pro-orchestrator-neon` → Planner: `gemini-4-neon`（`Gemini 4 Neon`），Executor: `gemini-4-hydrogen`（`Gemini 4 Hydrogen`）
- `gemini-pro-orchestrator` → Planner: `gemini-4-carbon`（`Gemini 4 Carbon`），Executor: `gemini-4-hydrogen`（`Gemini 4 Hydrogen`）
- `gemini-pro-orchestrator-argon` → Planner: `gemini-4-argon`（`Gemini 4 Argon`），Executor: `gemini-4-hydrogen`（`Gemini 4 Hydrogen`）

传给 `agy --settings` 的 `"orchestrator": {"planner": ..., "executor": ...}` 必须使用实时 `availableModels`（或标准标签格式）的准确展示名（例如 `"Gemini 4 Neon (XHigh)"`、`"Gemini 4 Hydrogen (Low)"`；注意 `xhigh` 对应 `"XHigh"`），而非 slug。若当前账号属于 Enterprise 档位，则仅保留原生暴露的选项；若后续 `agy acp` 原生为 Orchestrator 暴露 `thought_level`，GhostAP 会自动优先采用原生 ACP 配置。

GhostAP 使用 `model/effort`、`model/planner/executor` 或裸 `model` ID 作为持久选择标识；启动或在线切换时，如果后端拒绝选定模型或档位（如返回 `-32602` 或非法组合），将立即 fail-closed 报错，不静默回退到默认模型。

## 安全与隔离边界

- 认证复用本机 `agy` CLI 自身状态（例如 `~/.gemini/antigravity-cli`）；GhostAP 不读取、不复制也不硬编码用户凭据，也不改写用户的 `settings.json`。
- 子进程环境清理（`src/utils/env.py`）移除父级 Antigravity 会话注入的 `ANTIGRAVITY_AGENT`、`ANTIGRAVITY_LS_ADDRESS`、`ANTIGRAVITY_CSRF_TOKEN`、`ANTIGRAVITY_CONVERSATION_ID`、`ANTIGRAVITY_TRAJECTORY_ID`、`ANTIGRAVITY_SOURCE_METADATA`，避免从 AGY 会话内部派生子进程时串用父会话或语言服务器上下文。
- 原生 ACP 路径的权限请求通过 GhostAP 的 ACP `request_permission` 回调处理；仅 Orchestrator 自定义 Effort 的无头 CLI `stream-json` 路径传 `--dangerously-skip-permissions`（同其余自动化工具全授权执行契约）。

## Swarm 后台协作生命周期

AGY 原生 ACP `/swarm` 在首个 `session/prompt` 回合创建后台 worker 后即会先返回 `end_turn`，随后由后台 worker 通过共享消息板触发无显式 `session/prompt` 包装的反应式唤醒回合：

- **工作中状态与表情反馈**：`ACPSession` 与 `SwarmLifecycleTracker` 识别 `swarm`（`add` / `kill` / `status` / `model`）、`swarm_wait`、`post_message`、`read_board` 与 `exit_plan_mode` 信号；当检测到任务型 Swarm worker 启动或进入 `swarm_wait` 等待时，立即向原消息添加 `Fire`（`EmojiReaction.on_swarm_working()`）表情表示 Swarm 正在后台协作中，同时保持流式卡片、COT 过程流与 `RepoLock` 活跃，不因首轮 `end_turn` 提前结单或发 `PARTY`。
- **完整终态与完成通知**：在收到带 `[done]` 的最终汇报（或全部任务 worker 清理完成并给出最终总结）且短暂静默收敛后，GhostAP 才关闭主卡并添加 `PARTY`（`EmojiReaction.on_swarm_done()`）完成表情，同时发送一条简明飞书完成通知回复。
- **中断与空闲保护**：用户在等候期间执行 `/stop` 或卡片停止时，会立即唤醒并取消 Swarm 等待周期；后台等待期间 `UsageUpdate` 与 `request_permission` 帧也会刷新传输活动时间戳，避免误触发短空闲超时。
