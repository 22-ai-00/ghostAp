# DSH ACP 接入

## 运行拓扑

GhostAP 将 DSH 作为一等 ACP 编程后端启动：

```text
GhostAP -> dsh --profile acp -> @dsh-enhanced/acp -> DSH Agent
```

ACP 插件源码位于 `~/workspaces/githubgood/dsh-enhanced/plugins/acp`。本机开发环境使用
DSH profile 链接该目录：

```bash
cd ~/workspaces/githubgood/dsh-enhanced
pnpm build
dsh plugin --profile acp add ./plugins/acp
dsh plugin --profile acp list --depth 0
```

GhostAP 的可用性探测要求 `acp` profile 的依赖列表中存在
`@dsh-enhanced/acp`，避免仅安装了 `dsh` CLI 却在首次任务时才发现 profile
缺失。

## 用户入口

- `/dsh` 或 `/enter_dsh`：进入持久 DSH 编程模式。
- `/dsh_info`：查看当前会话与模型。
- `/model`：共享模型命令，自动适配当前 Codex、Traex、DSH 等 ACP 编程后端；
  在 DSH 模式中从实时 provider/model/reasoning effort 目录选择模型。
- `/exit`：共享退出命令，退出当前 DSH 或其他编程模式。
- Workflow Agent Pool、Deep 与 Spec 均可使用 `dsh`。

## 协议差异

DSH ACP 使用 namespaced config option：

- 模型：`dsh.model`，值为 `[provider, model]` JSON 数组。
- 推理等级：`dsh.reasoning_effort`，值为 `["default"]` 或
  `["effort", effort]`。

GhostAP 在模型发现时展开 DSH 的分组 provider 目录，并逐模型读取可用
reasoning effort；启动或热切换时再把持久选择还原为上述两个 ACP 请求。
模型不通过进程参数传递，因此 `dsh --profile acp` 始终保持稳定。

DSH 暴露的 `standard`、`code`、`minimal`、`cordis` 是 ACP session mode。
GhostAP 当前使用 profile 默认的 `standard`，不把这些预设伪装成不同工具。

## 安全边界

- GhostAP 不读取或复制 DSH provider 凭据，认证和 provider 配置归 DSH profile
  管理。
- 不启用自动批准；DSH 的权限请求继续进入 GhostAP ACP 一次性授权回调，保持
  fail-closed。
- ACP stdout 只承载协议帧；诊断由 DSH 插件写入 stderr。
- DSH `cordis` 预设具有更高进程内能力，GhostAP 不自动切换到该预设。
- Employee 隔离环境当前不会投影 `$DSH_HOME` profile 与凭据，因此雇佣入口不
  暴露 DSH；待有独立、最小权限的 profile 投影后再启用，避免形成运行时假支持。
