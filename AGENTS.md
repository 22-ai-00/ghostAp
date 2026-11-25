# AGENTS.md

这是 GhostAP 中 AI 编码代理的精简启动上下文（而非完整文档）。域内细节放入按需技能，历史记录放入 `.Memory/`。

## 项目概述

GhostAP 是一个通过出站 WebSocket 连接运行的飞书/Lark AI 研发网关：支持远程 Shell、项目管理与多模式 AI 编程，统一驱动 9 个可见工具（`coco`、`claude`、`aiden`、`codex`、`gemini`、`traex`、`grok`、`dsh`、`agy`）及 2 个隐藏 CLI 桥接（`claude_w`、`codex_w`）。

## 常用命令（仅使用 `uv`，禁 `pip`/`conda`）

```bash
uv sync --group dev
uv run python -m src.main --validate
uv run ruff check src/ tests/
uv run python -m pytest tests/ -q -m "not slow"
uv run python scripts/test_inventory.py tests/
GHOSTAP_LOG_MODE=append ./restart.sh
```

先跑最相关测试；修改共享路由、卡片、锁、配置或会话时跑全量快测。分层策略见 [docs/testing.md](docs/testing.md)。

## 核心契约与非显式陷阱

- **跨平台契约**：GhostAP 至少支持 Linux 与 macOS。涉及进程、文件锁、路径/临时目录、权限位、环境变量或平台工具时，必须用平台分支或能力探测保持两端契约并补回归；不得把 Linux-only 假设带到 macOS。
- **策略与状态隔离**：执行策略（普通编程 / Deep / Spec / Workflow）与传输（ACP / CLI）正交，新能力默认覆盖全部 11 后端。`SMART` 保留 shell 直通（含 `./restart.sh rr`）；`/coco`..`/agy` 设置持久聊天+项目状态；Deep、Spec 和 Workflow 是作用于飞书话题/根线程的引擎策略；它们不得替换聊天+项目编程状态。
- **并发与会话**：`ACPSessionManager` 统一解析会话密钥；任何原生 `threading.Lock()` / `RLock()` 必须按 [docs/adr-lock-ordering.md](docs/adr-lock-ordering.md) 标注 `# leaf lock: never held while acquiring a LockLevel lock`。
- **卡片边界与视觉验收**：严格保持 `handler -> session -> render / delivery` 单向导入（[docs/adr-card-pipeline-boundary.md](docs/adr-card-pipeline-boundary.md)）。改动 UI/卡片/HTML 前先更新 `ux/*.html` 预览，实现后渲染截图直接目检排版与截断。
- **安全与配置入口**：凭据仅经 `src/config/` 读 `.env`；仅当 `ADMIN_USER_IDS` 为空时主 Bot 私聊允许 `/setadmin` 单向引导。飞书斜杠面板 23 项由 `src/feishu/product_catalog.py` SSOT 投影，离板命令入 `HIDDEN_SLASH_COMMANDS`。
- **运行时自治 vs 编码协作**：产品运行时收到飞书任务后全自动推进至成功或明确失败终态，执行授权交给后端自身，不叠加策略过滤或确认门（[docs/product-contract.md](docs/product-contract.md)）。编码期间对齐周边代码风格与注释密度；改前先读 [.Memory/Abstract.md](.Memory/Abstract.md)，历史细节用 `rg .Memory/`；测试失败必查根因，临时脚本放 `tests/`、`scripts/`、`ux/` 或 `/tmp`；多阶段任务优先阶段 handoff 或分叉子代理而非中途压缩；独立修复在 `dev` 提交（[docs/commit-message-guidelines.md](docs/commit-message-guidelines.md)）后推 `dev` 再快进 `main`，并更新 `.Memory/{YYYY-MM-DD}.md`、滚动 `Abstract.md` 与 [.Memory/Backlog.md](.Memory/Backlog.md)。

## 按需加载技能（`.agents/skills/`）

按任务按需读取，不在首轮全量加载：

- [feishu-card-ui](.agents/skills/feishu-card-ui/SKILL.md) — Card JSON 2.0 规范、回调、`src/card/` 导入边界与视觉截图验收
- [workflow-engine](.agents/skills/workflow-engine/SKILL.md) — `/wf` 交互配置池、JS 运行时原语、超时终态与报告流水线
- [acp-backends](.agents/skills/acp-backends/SKILL.md) — 11 后端发现、原生 `/` 洁净透传、`/usage` & `/settings` 驾驶舱与会话生命周期
- [release-and-memory](.agents/skills/release-and-memory/SKILL.md) — `dev -> main` 发布、验证门禁、`./restart.sh` 排查、滚动 `.Memory/` 与阶段 handoff
