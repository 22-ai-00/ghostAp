# Grok ACP 编程模式实施计划

**目标：** 增加原生 Grok Build ACP 后端，使用户通过 `/grok` 进入与 `/codex` 等一致的持续编程模式，并可在 Workflow Agent Pool 中选择 Grok。

**架构：** 在 ACP provider 层封装 `grok agent [--model ...] stdio`，沿用统一 `ProgrammingModeHandler`、`ACPSessionManager`、卡片会话与项目上下文链路。路由、产品目录和帮助仅声明新的一等模式，不增加后端专属执行分支。

## 实施步骤

1. 先添加回归测试，锁定 provider 命令顺序、模式状态、意图识别、项目快照和产品目录合同，并确认测试因缺少 Grok 支持而失败。
2. 在 `src/acp/providers/` 注册 Grok provider，加入可用性探测和后台预热；保持 ACP 权限回调，不使用自动批准参数。
3. 在模式、项目上下文、处理器、分发器和 WebSocket 组装根中加入 Grok，使 `/grok`、`/exit_grok`、`/grok_info` 和模式内消息走统一编程卡片链路。
4. 更新卡片帮助、命令目录、Workflow 工具说明和 README，确保用户可发现 Grok。
5. 运行针对性测试、相关路由/卡片/ACP 扩展测试和项目校验；再执行本机 Grok ACP 初始化与模型发现的有界验证。
6. 更新 `.Memory`，按仓库提交规范提交并推送 `dev`。
