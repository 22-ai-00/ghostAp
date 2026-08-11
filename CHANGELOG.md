# Changelog

All notable changes to this project will be documented in this file.

> **另见**: [ADR: Pipeline 边界](docs/adr-card-pipeline-boundary.md)

## [Unreleased]

### Added

- **SessionLockPool**: Extracted from `CardDelivery` into `src/card/delivery/lock_pool.py`. Provides per-session RLock management with LRU eviction, TTL-based cleanup, and O(1) `drain()` via `threading.Condition` + in-flight counter.
- **TTLActuatorMixin**: Extracted from `CardSession` into `src/card/session/_ttl_mixin.py`. Encapsulates 15+ TTL actuator protocol methods (get_ttl_state, reduce_and_render, mark_ttl_expired, force_terminate, deliver_terminal, etc.).
- **Config validator**: `@field_validator` for `action_dedup_cleanup_interval` ensuring value in [1, 3600].
- **Domain partition comments** in `src/card/protocols.py` for clearer section navigation.

### Changed

- **官方飞书 SDK 升级**：`lark-channel-sdk` 从 1.1.0 升至 1.2.0，
  `lark-oapi` 从 1.7.1 升至 1.7.2；Employee Channel 的 wheel、RECORD
  与运行时 payload 完整性哈希同步重新锁定。
- **`BaseRenderer._create_session()` → `BaseRenderer.create_session()`**: Renamed to public API (no underscore prefix). All renderer subclasses and test patch targets updated.
- **`CardDelivery.drain_in_flight()`**: Now uses O(1) counter-based wait instead of O(n) lock iteration.

### ⚠️ Breaking Changes

- **飞书卡片回调仅支持最新版**：主 Bot 与 Employee Bot 只注册
  `card.action.trigger`，回调 ACK/toast/raw-card 更新统一使用官方 2.0 响应结构；
  出站交互卡只接受 `schema: "2.0"` + `body.elements`。旧
  `card.action.trigger_v1`、Card JSON 1.0、schema-less 卡片和根级
  `elements` 自动升级均已移除。部署时须在开放平台删除旧回调订阅并发布新版本。
- **Worktree mode removed**: `/worktree` and `/wt`, the Worktree engine, its cards, and its dedicated tests are gone. Existing historical cards are read-only and non-executable; use `/wf` as the supported multi-Agent orchestration entry. Legacy local branches and worktree directories are not deleted or migrated automatically.
- **TTADK backend removed**: `/ttadk` and `/enter_ttadk`, TTADK CLI/session startup, model selection, cards, project state, configuration, implementation modules, and dedicated tests are gone. Surviving programming tools continue through their existing ACP or CLI transports.
- **`DirectCardSession` 已删除**: `src/card/direct_session.py` 已移除。所有引擎渲染器现使用 `BaseRenderer.create_session()` → `CardSession.dispatch()`。
- **`CardBuilder.build_engine_card()` 已移除**: 访问将触发 `AttributeError`。使用新管线 `renderer.create_session()` + `session.dispatch(CardEvent.*)` 替代。
- **`src/card/adapter.py` 已删除**: 适配逻辑迁移至 `src/card/events/acp_adapter.py` 及 `src/card/protocols.py`。
- **`src/card/events.py` 顶层模块已删除**: 重构为 `src/card/events/` 包。
- **`BaseHandler` 旧方法已移除**: `reply_message`/`patch_message`/`send_message` 调用将 raise `NotImplementedError`。
- **Loop Engine 已完整移除**: `/loop`、`/loop_status`、`/loop_update`、`/stop_loop` 命令不再可用。Loop 模式的迭代工作流由 Spec 模式替代，多 Agent 编排由 Workflow 模式承接。相关源码（`src/loop_engine/`、`src/feishu/handlers/loop.py`、`src/feishu/renderers/loop_renderer.py`）及 15 个测试文件已删除。

### Removed

- **TTADK integration**: Removed `src/ttadk/`, its CLI session/wrapper layers, Feishu handler and callback surfaces, configuration/state fields, current documentation, UX references, and TTADK-specific tests.
- **`src/card/direct_session.py`** (`DirectCardSession`): Replaced by `CardSession` (`src.card.session.core`). All engine renderers now use `BaseRenderer.create_session()` → `CardSession.dispatch()`.
- **`src/card/adapter.py`**: Replaced by per-engine adapter functions in `src/card/events/acp_adapter.py` and engine-specific protocols in `src/card/protocols.py`.
- **`src/card/events.py`** (top-level module): Refactored into `src/card/events/` package (`types.py`, `factories.py`, `payloads.py`, `acp_adapter.py`).
- **`CardBuilder.build_engine_card()`**: 已完全移除（访问将触发 `AttributeError`）。Use `renderer.create_session()` + `session.dispatch(CardEvent.*)` instead.
- **Migration verification**: Run the docs/reference regression tests and `rg -n 'build_engine_card|DirectCardSession|_create_direct_session' src tests -g '*.py'` when touching the card migration boundary.
- **Deprecated card re-export shims**: Removed the old top-level `src/card/*` compatibility shims and their deadline checker. Canonical imports now live under `src/card/session/`, `src/card/delivery/`, `src/card/actions/`, and `src/card/timers/`.

### Migration FAQ

- **旧卡片上的按钮还能用吗？**
  已发送的旧卡片中的按钮将不再响应新操作（点击后会提示"按钮已失效"），但历史展示内容保持不变。请重新发送对应命令以获取新卡片。

- **升级后历史对话中的旧卡片会怎样？**
  历史对话中的旧卡片会保留为只读状态，显示内容不受影响，但交互功能（按钮、续期等）不再可用。新任务会自动使用新卡片格式。
