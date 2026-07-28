# Help Card Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 精简 `/help` 顶部格式，并将 `/wf_help` 从未渲染的纯文本改为合法、易读的飞书 Schema 2.0 卡片。

**Architecture:** `/help` 继续由 `SystemBuilder` 统一构建，只移除版本号和快捷按钮区域，不改变完整命令分区。`/wf_help` 复用 `CardBuilder._wrap_card()` 输出标准卡片，以 Markdown 摘要、前置要求和折叠分区组织内容，并由现有 `WorkflowHandler.reply_card()` 发送。

**Tech Stack:** Python 3.11、飞书卡片 Schema 2.0、pytest、HTML/CSS UX 预览。

## Global Constraints

- 仅使用 `uv` 运行 Python 与测试。
- UI 生产改动前先更新 `ux/card_preview.html`。
- 行为修改先写失败回归测试并确认红灯。
- 不改变 `/menu` 快捷菜单，也不改变 Workflow 命令语义。

---

### Task 1: 更新卡片 UX 预览

**Files:**
- Modify: `ux/card_preview.html`

**Interfaces:**
- Consumes: 用户提供的 `/help` 与 `/wf_help` 飞书截图。
- Produces: 无版本号、无“常用操作”按钮区的 `/help` 预览，以及结构化 `/wf_help` 卡片预览。

- [x] **Step 1: 删除 `/help` 预览中的版本号和快捷按钮区**

将标题改为 `📖 GhostAP 使用帮助`，并删除“常用操作”标题及按钮网格。

- [x] **Step 2: 添加 `/wf_help` 卡片预览**

增加 `⚡ Workflow 使用帮助` 卡片，包含简介、Node.js 前置要求、命令列表、执行流程和核心能力分区。

- [x] **Step 3: 检查预览结构**

Run: `rg -n "GhostAP 使用帮助|Workflow 使用帮助|常用操作|v0\\.2\\.0" ux/card_preview.html`

Expected: 新标题存在；Help 预览块中不再展示版本号与常用操作；Workflow 帮助预览存在。

### Task 2: 添加帮助卡片回归测试

**Files:**
- Modify: `tests/test_system_interaction.py`
- Modify: `tests/test_workflow_entry_help.py`

**Interfaces:**
- Consumes: `SystemHandler.show_full_help()` 与 `WorkflowHandler.show_workflow_help()`。
- Produces: `/help` 无版本号/按钮及 `/wf_help` 使用合法卡片的行为契约。

- [x] **Step 1: 修改 `/help` 断言**

断言标题严格等于 `📖 GhostAP 使用帮助`，卡片不包含按钮、`常用操作` 或旧版本标题。

- [x] **Step 2: 修改 `/wf_help` 断言**

断言 `reply_card()` 被调用、`reply_text()` 未调用，卡片为 Schema 2.0，标题与折叠分区正确且包含核心命令。

- [x] **Step 3: 运行测试确认红灯**

Run: `uv run python -m pytest tests/test_system_interaction.py::TestSystemInteraction::test_show_full_help tests/test_workflow_entry_help.py::TestWorkflowHelpText -q`

Expected: 旧实现因版本标题、快捷按钮和 `reply_text()` 路径而失败。

### Task 3: 实现 `/help` 与 `/wf_help` 卡片格式

**Files:**
- Modify: `src/card/ui_text.py`
- Modify: `src/card/builders/system.py`
- Modify: `src/feishu/handlers/workflow.py`

**Interfaces:**
- Consumes: `UI_TEXT`、`CardBuilder._wrap_card()`、`_node_version_required_text()`。
- Produces: 精简后的主帮助卡，以及由 `reply_card(message_id, card)` 发送的 Workflow 帮助卡。

- [x] **Step 1: 精简 `/help`**

把 `system_help_title` 改为无版本号标题，删除快速入口文案和构建逻辑，并清理只服务于该区域的私有 helper。

- [x] **Step 2: 构建 `/wf_help` Schema 2.0 卡片**

使用 turquoise header、Markdown 简介/前置要求、折叠命令/流程/能力分区；第一个分区默认展开，其余默认折叠。

- [x] **Step 3: 改用卡片发送**

`show_workflow_help()` 调用 `reply_card()`，不再把 Markdown 字符串交给 `reply_text()`。

- [x] **Step 4: 运行定向测试确认绿灯**

Run: `uv run python -m pytest tests/test_system_interaction.py tests/test_workflow_entry_help.py -q`

Expected: 全部通过。

### Task 4: 扩大验证与项目记忆

**Files:**
- Modify: `.Memory/2026-07-28.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: 本次 diff 与验证结果。
- Produces: 可追溯的实现、原因、验证和风险记录。

- [x] **Step 1: 运行卡片与 Workflow 相关测试**

Run: `uv run python -m pytest tests/test_card_builders.py tests/test_card_schema_contract.py tests/test_workflow_entry_help.py tests/test_system_interaction.py -q`

Expected: 全部通过。

- [x] **Step 2: 运行静态检查和配置校验**

Run: `uv run ruff check src/card/ui_text.py src/card/builders/system.py src/feishu/handlers/workflow.py tests/test_system_interaction.py tests/test_workflow_entry_help.py`

Run: `uv run python -m src.main --validate`

Run: `git diff --check`

Expected: 全部通过且无错误/警告。

- [x] **Step 3: 更新 Memory**

在 `.Memory/2026-07-28.md` 记录改动、根因、红绿测试、扩大验证和后续风险；在 `.Memory/Abstract.md` 添加约 20 字摘要及日期链接。
