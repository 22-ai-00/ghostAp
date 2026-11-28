# AGY ACP 与 Orchestrator CLI 混合接入

## 运行拓扑

GhostAP 将 Google Antigravity CLI（`agy`）作为一等编程后端，按所选模型配置自动路由到最合适的官方传输通道：

1. **常规模型与官方默认 Orchestrator 预设（原生 ACP stdio）**：
   ```text
   GhostAP -> agy acp (stdio JSON-RPC) -> Antigravity Agent
   ```
   适用于所有单个模型（含 `low`/`medium`/`high`/`xhigh` 原生 `thought_level`）以及默认预设的 `gemini-pro-orchestrator*`（`Planner Effort = default`），也保持完整 Native Swarm 生命周期、原生 `/` 命令、`request_permission` 与 `is_cli_backend("agy") == False`。
2. **Orchestrator 自定义 Planner/Executor 模型与 Effort（CLI `stream-json`）**：
   ```text
   GhostAP -> agy --prompt=... --output-format stream-json --dangerously-skip-permissions --model <orch_id> --settings <json> [--conversation <id>]
   ```
   `agy acp`（1.5.18）尚未对 `gemini-pro-orchestrator*` 原生暴露 `thought_level` 或按会话隔离的 `orchestrator` 配置项；因此在选择自定义 `<orch_id>/<planner_effort>/<executor_effort>` 或 `<orch_id>/<planner_model>:<p_eff>/<executor_model>:<e_eff>` 时，GhostAP 通过每进程独立 `--settings '{"showThinking": true, "orchestrator": {"planner": "<Planner Display Label>", "executor": "<Executor Display Label>"}}'` 启动并跨轮以 `--conversation` 续接会话，不篡改宿主 `~/.gemini/antigravity-cli/settings.json`。

可用性探测仍通过 `agy acp --help` 检查 `usage:`、`agy acp` 与 `stdio` 关键字，确保在服务就绪与工具面板阶段即可验证当前主机安装的 `agy` CLI 支持 ACP stdio server。

## 用户入口

- `/agy` 或 `/enter_agy`：进入持久 AGY 编程模式。
- `/agy_info`：查看当前 AGY 会话与模型/Orchestrator Effort 信息。
- `/model`：共享模型命令，自动适配当前 Codex、Traex、DSH、AGY 等编程后端；
  - 在 AGY 模式中实时探测 `model`、`thought_level`（或 `effort`）与 `availableModels` 展示标签，并将底层 `gemini-pro-orchestrator*` 原生预设收拢为单一 **🪐 Orchestrator（双阶段规划+执行）** 入口，展开四维受限选择器：
    - **Planner Model**（严格限定 4 种）：`gemini-4-neon`、`gemini-4-carbon`（默认）、`gemini-4-argon`、`claude-opus-5-5`
    - **Planner Effort**：所选 Planner 模型的实时 `reasoning_efforts`；当 Executor 为 `gemini-4-hydrogen` 且目录中存在对应原生预设时额外提供 `default`（原生 ACP 预设，支持 `/swarm`）。`claude-opus-5-5` 无对应原生预设，不提供 `default`
    - **Executor Model**（严格限定 3 种）：`gemini-4-hydrogen`（默认）、`gemini-4-neon`、`claude-sonnet-5-5`
    - **Executor Effort**：所选 Executor 模型的实时 `reasoning_efforts`（原生 `default` 模式下跟随显示 `default`）
  - 初次打开未保存选择的 `/agy` 选型卡默认预选 `gemini-pro-orchestrator/high/high`（同时在 Planner Effort 中保留 `default` 供切换）；若当前已保存原生预设（如 `gemini-pro-orchestrator` / `gemini-pro-orchestrator-neon`），再次打开则保留在 `default` 原生预设态。
  - `/model <slug>` 同时兼容 AGY CLI 连字符别名（如 `gemini-4-neon-xhigh` → `gemini-4-neon/xhigh`）、`orchestrator` / `orchestrator/<rest>` 简写（映射至 `gemini-pro-orchestrator`）与 Orchestrator 两段简写（如 `gemini-pro-orchestrator-neon/xhigh`）。
- `/effort`：共享 Effort 查看与切换命令（例如 `/effort xhigh`、`/effort xhigh low`、`/effort xhigh/low`、`/effort default`）。
- `/exit`：共享退出命令，退出当前 AGY 或其他编程模式。
- Deep、Spec 与 Workflow Agent Pool 均可使用 `agy`；跨原生 ACP 与 CLI 自定义 Orchestrator 切换模型/Effort 时，GhostAP 会自动原子替换底层会话传输，同一传输内则直接热更新。

## 协议细节

### 原生 ACP（常规模型与默认 Orchestrator）

`agy acp` 不通过 CLI 参数传入 `--model`，而是始终以无模型参数方式启动 `agy acp`，并在 `session/new` 之后通过 ACP `session/set_config_option` 应用模型选择：

- 模型：`configId="model"`，选项来自实时 `session/new` 返回的 `configOptions`（如 `gemini-4-neon`、`gemini-4-carbon`、`gemini-4-argon`、`gemini-4-hydrogen`、`gemini-pro-orchestrator-neon`、`gemini-pro-orchestrator`、`gemini-pro-orchestrator-argon`、`claude-opus-5-5`、`claude-sonnet-5-5`、`gpt-oss-120b-medium` 等）。
- 思考档位：`configId="thought_level"`（兼容 `reasoning_effort`/`effort`），由 GhostAP 在模型发现阶段逐模型切换 `model` 并读取对应的实时档位选项（例如 `gemini-4-neon` / `gemini-4-hydrogen` 支持 `low`/`medium`/`high`/`xhigh`，`gemini-4-carbon` / `claude-opus-5-5` / `claude-sonnet-5-5` 支持 `low`/`medium`/`high`，而 `gpt-oss-120b-medium` 不暴露思考档位）。

### 统一 Orchestrator 模式与严格角色白名单（消费级实时合成 + CLI `stream-json`）

在 `agy` 1.5.18 中，Orchestrator 在交互卡片中收拢为单一模式入口（优先映射到 `gemini-pro-orchestrator`），不再拆分为 3 个平行顶层模型项，并在解析、组合与目录校验全链路严格执行 fail-closed 白名单约束：

- **允许的 Planner 模型（4 种）**：`gemini-4-neon`、`gemini-4-carbon`、`gemini-4-argon`、`claude-opus-5-5`
- **允许的 Executor 模型（3 种）**：`gemini-4-hydrogen`、`gemini-4-neon`、`claude-sonnet-5-5`
- **原生 ACP 预设映射（仅当 Executor 为 `gemini-4-hydrogen` 且 Planner Effort 为 `default` 时可用）**：
  - `gemini-4-neon` → `gemini-pro-orchestrator-neon`（Planner: `Gemini 4 Neon`，Executor: `Gemini 4 Hydrogen`）
  - `gemini-4-carbon` → `gemini-pro-orchestrator`（Planner: `Gemini 4 Carbon`，Executor: `Gemini 4 Hydrogen`）
  - `gemini-4-argon` → `gemini-pro-orchestrator-argon`（Planner: `Gemini 4 Argon`，Executor: `Gemini 4 Hydrogen`）
  - `claude-opus-5-5` 无原生预设；选择自定义 Executor（如 `gemini-4-neon`、`claude-sonnet-5-5`）时同样必须走自定义 Effort 组合。
- **Fail-closed 校验**：任何非白名单角色模型（如 `gpt-oss-120b-medium`、`gemini-3.8-flash`、将 `gemini-4-hydrogen` 用作 Planner、或将 `gemini-4-carbon`/`claude-opus-5-5` 用作 Executor）以及缺失 Planner/Executor Effort 的自定义组合均会在 `parse_agy_model_selection`、`compose_agy_orchestrator_selection`、`is_valid_agy_catalog_selection` 与 `validate_agent_binding_spec` 中被直接拒绝，不会静默回退。

传给 `agy --settings` 的 `"orchestrator": {"planner": ..., "executor": ...}` 必须使用实时 `availableModels`（或标准标签格式）的准确展示名（例如 `"Gemini 4 Neon (XHigh)"`、`"Gemini 4 Hydrogen (Low)"`、`"Claude Opus 5.5 (High)"`；注意 `xhigh` 对应 `"XHigh"`），而非 slug。若当前账号属于 Enterprise 档位，则仅保留原生暴露的选项；若后续 `agy acp` 原生为 Orchestrator 暴露 `thought_level`，GhostAP 会自动优先采用原生 ACP 配置。

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
