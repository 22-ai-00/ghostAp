# Workflow Agent Pool 去重与性能修复实施计划

> **执行方式：** 当前共享工作区内按 TDD 顺序实施；每个行为先验证 RED，再写最小生产改动。计划存放于仓库允许的普通 `docs/` 目录。

**目标：** 不同 `tool/model/profile/effort` 的 Agent 能连续加入同一 WF Pool，真实快速重放仍被拦截；模型目录热路径不重复探测或重复解析。

**架构：** 保留 Card JSON 2.0 的工具/模型族/Profile/Effort 级联，不恢复已触发 `200530` 的混合 form。卡片渲染把当前 draft/pool 状态签名写入 Add callback，通用消息缓存按命中的具体 key 精确执行 TTL。ACP 目录继续以 `(tool, cwd)` 为权威 key，但把正/负缓存窗口延长，并让一次 Workflow action 复用同一个解析结果。

**技术栈：** Python 3.13、pytest、Pydantic Workflow 状态、飞书 Card JSON 2.0、ACP model catalog。

## 全局约束

- 仅使用 `uv`；不使用 pip/conda。
- 不移除 Workflow 的模型族/Profile/Effort 级联。
- 不重新引入 CardKit form 或 `form_submit`。
- Agent Pool 仍只拒绝完全相同的最终 `tool + composed model selection`。
- 卡片 action 去重仍保留；只有状态不同或 TTL 已到期的合法操作放行。
- 所有功能和 bug 修复先有可观察的失败测试。

---

### Task 1：修复合法 Add 被入口去重误拦

**文件：**

- 修改：`src/workflow_engine/renderer.py`
- 修改：`src/feishu/message_cache.py`
- 修改：`src/card/ui_text.py`
- 测试：`tests/test_workflow_agent_selection_contract.py`
- 测试：`tests/test_button_gate_and_dedupe.py`
- 测试：`tests/test_message_cache.py`

**接口：**

- 产出：`workflow_add_agent` callback value 中稳定的 `_selection_sig`。
- 保持：`MessageCache.is_duplicate(key) -> bool` 公共签名不变。

- [x] 写 renderer 回归：相同 draft/pool 的签名稳定，Codex 与 Traex draft 的签名不同。
- [x] 写入口回归：同一 card/action 的两个不同 `_selection_sig` 都提交；第二次相同签名才去重。
- [x] 写 fake-clock 回归：`ttl=1, cleanup_interval=30` 时，同 key 在 1 秒后立即视为新事件。
- [x] 运行上述测试并确认分别因缺签名、过期 key 仍命中而 RED。
- [x] 在 renderer 中以规范化 draft 与 pool 生成短 SHA-256 状态签名，仅作为 UI/去重内部字段。
- [x] 在 `MessageCache.is_duplicate()` 命中具体 key 时先检查该条目的 TTL；过期则删除并重新登记。
- [x] 将重复事件 toast 改为中性的“操作正在处理中，请稍候”，避免暗示用户选择重复。
- [x] 运行三组测试并确认 GREEN。

### Task 2：延长模型目录缓存并消除单次 action 重复读取

**文件：**

- 修改：`src/acp/helper.py`
- 修改：`src/feishu/handlers/workflow.py`
- 创建：`tests/test_acp_model_cache.py`
- 测试：`tests/test_workflow_agent_selection_contract.py`

**接口：**

- 保持：`fetch_acp_models(tool_name, cwd, current_model=None, probe_timeout=None)` 签名不变。
- 扩展：`_workflow_selection_card_data(..., model_state=...)` 可消费本次 action 已解析的状态。

- [x] 写 fake-clock 正缓存回归：普通 ACP catalog 在 30 分钟内只探测一次。
- [x] 写 fake-clock 负缓存回归：失败 catalog 在 5 分钟内不重复启动 provider。
- [x] 写 Workflow 回归：模型/Profile/Effort/显式 Add 每次 action 最多调用一次 `fetch_acp_models`。
- [x] 运行测试并确认当前 5 分钟/45 秒窗口及双调用导致 RED。
- [x] 将普通 ACP 正缓存改为 1800 秒、负缓存改为 300 秒，保留显式 exact-key invalidation。
- [x] 把 action 内已得到的 `ModelCascadeState` 传给重绘，避免验证后再次调用 helper。
- [x] 将 Coco 目录同步为 30 分钟正缓存、5 分钟 fallback 缓存，并接入通用显式失效。
- [x] 用 generation fence 阻止失效前探测回写，并关闭 miss/flight 注册之间的 TOCTOU 重探测。
- [x] 让非法 callback 与 stale Add 的卡面、服务端 draft 和状态签名保持一致且可恢复。
- [x] 运行目标测试并确认 GREEN。

### Task 3：扩大验证与项目记忆

**文件：**

- 修改：`.Memory/2026-08-11.md`
- 修改：`.Memory/Abstract.md`

- [x] 运行最相关测试文件。
- [x] 运行全部 Workflow、卡片去重与 ACP 模型相关回归。
- [x] 运行 `uv run ruff check`（触达文件）、`uv run python -m src.main --validate` 和 `git diff --check`。
- [x] 如共享路由/缓存改动通过聚焦与扩大回归，再运行非慢全库测试。
- [x] 在当日 Memory 记录根因、改动、实测数据、验证和剩余外部真实飞书验收风险；Abstract 增加一行索引。
