# Workflow 卡片重复推送修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Workflow 生成与执行阶段因卡片 provenance 丢失和心跳未收口而持续新建或刷新飞书卡片的问题。

**Architecture:** 保留 main Bot outbound audit 的默认拒绝策略，将用户入站消息的可信 origin 从首张选择卡贯穿生成卡、进度卡和终态 fallback 卡。周期性进度只允许 PATCH 既有卡片，不能在持续 PATCH 失败时反复 CREATE。生成、卡片投递和排队启动共享 session-keyed owner，并通过 engine lock 下的 CAS 与 `/stop_wf` 线性化；终态投递与在途进度 PATCH 串行。

**Tech Stack:** Python 3.13、threading、pytest、unittest.mock、飞书 Card 2.0、GhostAP MessageLinker。

## Global Constraints

- 仅使用 `uv` 运行 Python 和测试。
- 不放宽 `main Bot patch recipient scope` 的 fail-closed 审计。
- 一个 Workflow 用户任务只保留一张活动主卡；周期性进度 PATCH 失败不得创建替代卡，只有首次/终态等有界交付允许创建一次继承可信 origin 的替代卡。
- 所有 heartbeat 在终态、异常、无效模板或显式停止后都不得继续投递。
- 先运行最相关回归，再扩大到 Workflow、卡片、配置/验证和完整发布门禁。

---

### Task 1: 锁定 provenance 丢失导致的 fallback 新卡循环

**Files:**
- Modify: `tests/test_workflow_confirm.py`
- Modify: `tests/test_workflow_orchestrator_select.py`

**Interfaces:**
- Consumes: `WorkflowHandler._build_workflow_callbacks(message_id, chat_id, project)` 和 `BaseHandler.send_card_to_chat(..., origin_message_id=...)`。
- Produces: 回归契约，证明生成卡与运行期 fallback 卡都携带同一个可信 origin。

- [x] **Step 1: 为周期性进度失败写失败测试**

把 `handler._resolve_origin` 固定为返回 `"origin_msg"`，连续两次令 PATCH
失败，并断言两次都只更新同一个 message ID，且：

```python
handler.send_card_to_chat.assert_not_called()
```

另加终态 fallback 回归，证明有界替代卡仍携带
`origin_message_id="origin_msg"`。

- [x] **Step 2: 为首张选择卡与生成卡写失败测试**

使用真实 `MessageLinker` 验证首张选择卡、终态 fallback 和后续 PATCH
均能反查原始 `/wf` 消息；生成并自动执行的回归中固定：

```python
handler._resolve_origin = MagicMock(return_value="origin_msg")
```

并断言 loading 卡通过以下调用创建：

```python
handler.send_card_to_chat.assert_called_once_with(
    "test_chat",
    ANY,
    origin_message_id="origin_msg",
)
```

- [x] **Step 3: 运行红灯**

Run:

```bash
uv run python -m pytest \
  tests/test_workflow_confirm.py::TestWorkflowHandlerConfirmFlow::test_workflow_callbacks_do_not_create_new_cards_when_progress_patch_keeps_failing \
  tests/test_workflow_orchestrator_select.py::test_generate_and_show_confirm_card_auto_starts_without_confirm_card \
  -q
```

Expected: 原实现分别因周期性进度创建新卡、首卡/后续卡丢失 origin 而失败。

---

### Task 2: 锁定生成 heartbeat 的异常与停止生命周期

**Files:**
- Modify: `tests/test_workflow_orchestrator_select.py`
- Modify: `tests/test_workflow_state_consistency.py`

**Interfaces:**
- Consumes: `WorkflowHandler._generate_and_show_confirm_card(...)` 与 `WorkflowHandler.stop_workflow(...)`。
- Produces: 无效模板/异常返回和显式停止后 heartbeat stop event 已设置、线程已 join 的契约。

- [x] **Step 1: 写无效模板后的失败测试**

用无真实等待的 fake thread 捕获 `target`、`start()` 和 `join()`；执行已有无效模板路径后断言：

```python
assert fake_thread.started is True
assert fake_thread.joined is True
assert fake_thread.target_stop_event.is_set()
```

测试应验证生产方法返回后不再存在可继续触发 `_heartbeat_update()` 的线程。

- [x] **Step 2: 写 `/stop_wf` 的失败测试**

为 `GENERATING_SCRIPT` engine 绑定一个未设置的生成取消 event，执行：

```python
handler.stop_workflow("msg_1", "chat_1", project)
```

随后断言 event 已设置、状态为 `IDLE` 且 pending 已清理。

- [x] **Step 3: 运行红灯**

Run:

```bash
uv run python -m pytest \
  tests/test_workflow_orchestrator_select.py \
  tests/test_workflow_state_consistency.py \
  -q
```

Expected: 新增 lifecycle 断言失败，既有断言继续通过。

---

### Task 3: 贯穿可信 origin 并收口生成 heartbeat

**Files:**
- Modify: `src/feishu/handlers/workflow.py`
- Modify: `src/feishu/handlers/workflow_selection.py`
- Modify: `src/workflow_engine/engine.py`

**Interfaces:**
- Consumes: `BaseHandler._resolve_origin(message_id) -> str` 和 `send_card_to_chat(..., origin_message_id=...)`。
- Produces: 每个 Workflow 替代卡都可由 MessageLinker 反查原消息；生成 heartbeat 在所有退出路径停止。

- [x] **Step 1: 修复首卡与生成卡 origin**

在 `_generate_and_show_confirm_card` 创建 loading 卡前解析：

```python
origin_message_id = self._resolve_origin(message_id)
```

并使用：

```python
gen_msg_id = self.send_card_to_chat(
    chat_id,
    gen_card,
    origin_message_id=origin_message_id,
)
```

- [x] **Step 2: 修复替代卡 origin 并限制周期性 fallback**

给 `_replace_or_send_workflow_card` 和
`_replace_or_send_workflow_rendered_card` 增加可选
`origin_message_id` 参数；PATCH 失败时通过：

```python
return self.send_card_to_chat(
    chat_id,
    card,
    origin_message_id=origin_message_id,
)
```

`_show_initial_workflow_progress_card` 和 `_build_workflow_callbacks` 在首个
PATCH 前解析一次可信 origin，并在后续所有进度/终态 fallback 中复用。
运行期 heartbeat/progress 传 `fallback_to_new=False`。

- [x] **Step 3: 用 finally 关闭生成 heartbeat**

把 engine 获取、模板发现和脚本生成放入：

```python
try:
    ...
finally:
    _stop_heartbeat()
```

删除仅在正常路径执行的裸 `_stop_heartbeat()`，保证 return 和异常同样关闭。

- [x] **Step 4: 让显式停止触发生成取消 event**

当前生成会话在 engine 上登记 session-keyed owner；`stop_workflow()`
在同一 engine lock 下设置 stop event、清理状态和 owner，再等待
delivery lock 后回复。生成函数只清理自己登记的 owner。

- [x] **Step 5: 让生成提交与排队启动可线性化**

生成结果通过 `_commit_generation_result_if_current()` 在 engine lock
下 CAS；排队执行通过独立 start owner 在 `execute_workflow()` 入口再次
原子 claim，停止获胜时不得清除 cancel event 或发布 RUNNING。

- [x] **Step 6: 串行进度与终态投递**

进度 callback 持 delivery lock 完成状态检查和 PATCH；done/error 等待
在途 PATCH 后设置 terminal fence，再生成报告和投递终态卡。

- [x] **Step 7: 实现 newest-wins 与唯一脚本 ownership**

同一发起者或管理员的新请求原子撤销旧 selection、generation 和未 claim
queued start；已 claim runtime 仍需显式停止。生成源文件和确认执行副本均
使用 owner 唯一路径，CAS 输家只删除自己的产物。

- [x] **Step 8: 等待 bridge 与 lifecycle worker 真正静默**

RuntimeBridge stop 后关闭提交门并等待 agent/subworkflow future；
WorkflowEngine 只在 bridge、AgentExecutor 和 owner 全部完成后发布 IDLE。
cleanup 超时保留 STOPPING tombstone，并用 retired-owner 集合跨重试追踪
started/preclaim worker。

- [x] **Step 9: 线性化选择状态与卡片渲染**

所有选择 mutation、finish snapshot 和状态迁移统一使用 engine → selection
锁序；legacy confirm 在相同 CAS 内捕获工具列表。交付前比较 render snapshot，
阻止慢旧卡覆盖较新同会话卡。

- [x] **Step 10: 运行绿灯**

Run:

```bash
uv run python -m pytest \
  tests/test_workflow_confirm.py \
  tests/test_workflow_orchestrator_select.py \
  tests/test_workflow_state_consistency.py \
  tests/test_workflow_heartbeat.py \
  tests/test_workflow_progress_coalescer.py \
  -q
```

Expected: 全部通过，且没有未退出线程或额外摘要错误。

---

### Task 4: 扩大验证并记录项目记忆

**Files:**
- Modify: `.Memory/2026-07-31.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: 修复后的 Workflow handler 和回归证据。
- Produces: 可追溯的根因、改动、验证结果与剩余外部验收风险。

- [x] **Step 1: 运行 Workflow 与卡片相关套件**

Run:

```bash
uv run python -m pytest tests/test_workflow*.py -q
uv run ruff check src/feishu/handlers/workflow.py \
  src/feishu/handlers/workflow_selection.py \
  src/workflow_engine/engine.py \
  tests/test_workflow_confirm.py \
  tests/test_workflow_orchestrator_select.py \
  tests/test_workflow_state_consistency.py
uv run python -m src.main --validate
git diff --check
```

Expected: 全部退出 0。

- [x] **Step 2: 运行完整发布门禁**

Run:

```bash
uv run python -m pytest tests/ -q -m "not slow"
uv run python -m pytest tests/test_workflow*.py -q
```

Result: 全仓非慢层 12,664 passed；Workflow 完整层 929 passed。
首次全仓门禁发现 leaf-lock 注释扫描失败，补齐约定注释后整轮重跑通过。

- [x] **Step 3: 更新 Memory**

在 `.Memory/2026-07-31.md` 记录：

```markdown
## Workflow 卡片重复推送与生命周期竞态修复

### 现场与根因
...

### 变更
...

### 验证
...

### 后续风险
...
```

并在 `.Memory/Abstract.md` 的 `2026-07-31` 下添加约 20 字摘要和日期链接。

---

### Task 5: 复核、提交并推送 dev

**Files:**
- Modify: 仅以上计划、生产代码、测试与 Memory 文件。

**Interfaces:**
- Consumes: 全绿验证和干净的变更范围。
- Produces: `origin/dev` 上可复现的修复提交。

- [x] **Step 1: 复核范围**

Run:

```bash
git status --short
git diff --stat
git diff
git diff --check
```

Expected: 无无关文件、凭据、日志或临时产物进入提交。

- [x] **Step 2: 按规范提交**

```bash
git add \
  src/engine_base.py \
  src/feishu/handlers/workflow.py \
  src/feishu/handlers/workflow_selection.py \
  src/workflow_engine/bridge.py \
  src/workflow_engine/engine.py \
  src/workflow_engine/manager.py \
  tests/test_engine_base_di.py \
  tests/test_workflow_bridge_transport.py \
  tests/test_workflow_confirm.py \
  tests/test_workflow_lifecycle_linearization.py \
  tests/test_workflow_orchestrator_select.py \
  tests/test_workflow_queued_start_cleanup.py \
  tests/test_workflow_save_reuse.py \
  tests/test_workflow_startup_fixes.py \
  tests/test_workflow_state_consistency.py \
  docs/2026-07-31-workflow-card-flood-fix-plan.md \
  .Memory/2026-07-31.md \
  .Memory/Abstract.md \
  .Memory/Backlog.md
git commit -m "fix(workflow): prevent card floods and stale runs"
```

- [x] **Step 3: 推送 dev**

```bash
git fetch origin dev
git push origin dev
```

Expected: 本地 `dev` 与 `origin/dev` 指向同一新提交。
