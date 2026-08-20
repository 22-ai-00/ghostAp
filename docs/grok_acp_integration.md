# Grok Build ACP 接入调研

## 结论

xAI 官方的 Grok Build CLI 原生实现了 Agent Client Protocol（ACP），无需第三方适配器。GhostAP 应通过标准输入输出启动：

```text
grok agent stdio
```

当用户选择了模型时，Grok 的全局模型参数必须放在 `stdio` 子命令之前：

```text
grok agent --model <model> stdio
```

## 官方依据

- [xAI Grok Build 官方仓库](https://github.com/xai-org/grok-build)：README 明确说明可通过 ACP 嵌入 Grok Build，并给出官方 CLI 安装方式。
- [Grok Build 官方概览](https://docs.x.ai/build/overview)：说明 Grok Build 的定位与 CLI 使用方式。
- [Grok Build 官方设置](https://docs.x.ai/build/settings)：说明 `--model`、默认模型 `grok-build`、`GROK_HOME` 和 `XAI_API_KEY` 等配置。
- [Grok Build 权限与安全](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-pager/docs/user-guide/22-permissions-and-safety.md)：记录 ACP stdio 启动及权限选项。
- [Grok Build 企业部署](https://docs.x.ai/build/enterprise)：记录设备授权和 API Key 认证方式。

## 本机探测

本机已安装 `/home/jiataorui/.local/bin/grok`，版本为 `1.0.0 (3cd0d0cbce)`；`grok agent --help` 提供 `stdio` 子命令以及 `--model`、`--reasoning-effort` 等选项。

## 权限与兼容决策

- GhostAP 不叠加 ACP 工具过滤或命令风险判断；权限回调选择后端提供的允许项，实际执行权限由 Grok 自身定义。
- 认证交由官方 CLI 管理；GhostAP 不读取、不复制、不硬编码凭据。
- 模型列表由 ACP `config_options` 动态发现；未显式选择时使用 Grok 后端默认模型。
- Grok 作为第七个一等编程后端接入普通编程模式，同时进入 Workflow 可用工具池。
