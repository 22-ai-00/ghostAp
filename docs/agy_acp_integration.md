# AGY ACP 接入

## 运行拓扑

GhostAP 将 Google Antigravity CLI（`agy`）作为一等 ACP 编程后端启动，且仅通过原生 ACP 协议对接（不使用 shell CLI 桥接或 `--dangerously-skip-permissions`）：

```text
GhostAP -> agy acp (stdio JSON-RPC) -> Antigravity Agent
```

可用性探测通过 `agy acp --help` 检查 `usage:`、`agy acp` 与 `stdio` 关键字，确保在服务就绪与工具面板阶段即可验证当前主机安装的 `agy` CLI 支持 ACP stdio server。

## 用户入口

- `/agy` 或 `/enter_agy`：进入持久 AGY 编程模式。
- `/agy_info`：查看当前 AGY 会话与模型信息。
- `/model`：共享模型命令，自动适配当前 Codex、Traex、DSH、AGY 等 ACP 编程后端；
  在 AGY 模式中实时探测 `model` 与各模型对应的 `thought_level` 能力，生成模型/思考档位级联下拉。
- `/exit`：共享退出命令，退出当前 AGY 或其他编程模式。
- Deep、Spec 与 Workflow Agent Pool 均可使用 `agy`。

## 协议细节

`agy acp` 不通过 CLI 参数传入 `--model`，而是始终以无模型参数方式启动 `agy acp`，并在 `session/new` 之后通过 ACP `session/set_config_option` 应用模型选择：

- 模型：`configId="model"`，选项来自实时 `session/new` 返回的 `configOptions`（如 `gemini-4-carbon`、`gemini-4-hydrogen`、`gemini-pro-orchestrator`、`claude-opus-5-5`、`gpt-oss-120b-medium` 等）。
- 思考档位：`configId="thought_level"`，由 GhostAP 在模型发现阶段逐模型切换 `model` 并读取对应的实时 `thought_level` 选项（例如 `gemini-4-hydrogen` 支持 `low` / `medium` / `high` / `xhigh`，而 `gemini-pro-orchestrator` 与 `gpt-oss-120b-medium` 不暴露 `thought_level`）。

GhostAP 使用 `model/effort`（当存在思考档位时）或裸 `model` ID 作为持久选择标识；启动或在线热切换时，如果后端拒绝选定模型或档位（如返回 `-32602`），将立即 fail-closed 报错，不静默回退到默认模型。

## 安全与隔离边界

- 认证复用本机 `agy` CLI 自身状态（例如 `~/.gemini/antigravity-cli`）；GhostAP 不读取、不复制也不硬编码用户凭据。
- 子进程环境清理（`src/utils/env.py`）移除父级 Antigravity 会话注入的 `ANTIGRAVITY_AGENT`、`ANTIGRAVITY_LS_ADDRESS`、`ANTIGRAVITY_CSRF_TOKEN`、`ANTIGRAVITY_CONVERSATION_ID`、`ANTIGRAVITY_TRAJECTORY_ID`、`ANTIGRAVITY_SOURCE_METADATA`，避免从 AGY 会话内部派生子 `agy acp` 进程时串用父会话或语言服务器上下文。
- 权限请求继续通过 GhostAP 的 ACP `request_permission` 回调处理（采用后端提供的允许项，不传递 `--dangerously-skip-permissions`）。
- Employee 隔离环境当前仅替换员工 `HOME` 而不投影 `~/.gemini/antigravity-cli` 认证状态，因此数字员工 `/hire` 入口暂不暴露 `agy`，避免形成运行时假支持。

## Swarm 后台协作生命周期

AGY 原生 `/swarm` 在首个 `session/prompt` 回合创建后台 worker 后即会先返回 `end_turn`，随后由后台 worker 通过共享消息板触发无显式 `session/prompt` 包装的反应式唤醒回合：

- **工作中状态与表情反馈**：`ACPSession` 与 `SwarmLifecycleTracker` 识别 `swarm`（`add` / `kill` / `status` / `model`）、`swarm_wait`、`post_message`、`read_board` 与 `exit_plan_mode` 信号；当检测到任务型 Swarm worker 启动或进入 `swarm_wait` 等待时，立即向原消息添加 `Fire`（`EmojiReaction.on_swarm_working()`）表情表示 Swarm 正在后台协作中，同时保持流式卡片、COT 过程流与 `RepoLock` 活跃，不因首轮 `end_turn` 提前结单或发 `PARTY`。
- **完整终态与完成通知**：在收到带 `[done]` 的最终汇报（或全部任务 worker 清理完成并给出最终总结）且短暂静默收敛后，GhostAP 才关闭主卡并添加 `PARTY`（`EmojiReaction.on_swarm_done()`）完成表情，同时发送一条简明飞书完成通知回复。
- **中断与空闲保护**：用户在等候期间执行 `/stop` 或卡片停止时，会立即唤醒并取消 Swarm 等待周期；后台等待期间 `UsageUpdate` 与 `request_permission` 帧也会刷新传输活动时间戳，避免误触发短空闲超时。
