# ACP Provider 接入指南

## 概述

本指南说明如何在 GhostAP 系统中新增一个编程后端（ACP 协议或 CLI 桥接模式），并接入共享的后端单一事实来源（SSOT）注册表。

## 架构概览

系统已提供声明式 ACP 适配与全链路 SSOT 元数据层：

1. **ACPProvider Protocol**（`src/acp/provider.py`） - 统一工具探测与启动命令接口
2. **ToolRegistry**（`src/acp/provider.py`） - 工具注册、可用性缓存与后台预热中心
3. **SyncACPSession**（`src/acp/sync_adapter.py`） - 同步 ACP 会话管理与生命周期治理
4. **Backend SSOT**（`src/mode/backends.py`、`src/card/backend_meta.py`） - 编程后端元数据单一事实来源，自动派生模式、命令目录、意图识别、上下文快照、会话管理器与卡片样式
5. **传输边界** - ACP Provider 与 shell CLI 后端（`transport="acp"` vs `"cli"`）保持独立，只有声明支持 ACP 的后端才会启动 ACP Server

## 接入步骤

### 1. 实现并注册 Provider（`src/acp/providers/` + `src/acp/providers/__init__.py`）

所有标准与定制 ACP Provider 均在 `src/acp/providers/` 下实现并在 `src/acp/providers/__init__.py` 的 `_ensure_providers()` 中延迟构建并注册到 `tool_registry`：

- **常规 CLI / stdio 工具**：直接在 `src/acp/providers/__init__.py` 的 `_ensure_providers()` 中添加探活 checker（如 `_make_custom_help_checker_with_cache_handle`、`_make_probe_checker_with_cache_handle` 或 `_make_resolve_checker`）与 `_ProviderConfig`，由 `GenericACPProvider` 统一承载并调用 `tool_registry.register(...)`。
- **需要自定义命令或模型规范化的工具**：在 `src/acp/providers/` 下继承 `GenericACPProvider`（或实现 `ACPProvider` 协议，参考 `src/acp/providers/__init__.py` 中的 `CodexACPProvider`、`TraexACPProvider`、`GrokACPProvider`），并在 `_ensure_providers()` 中实例化注册（如需外部导入可同步导出至 `__all__`）：

```python
from .mytool import MyToolACPProvider  # 或在 src/acp/providers/__init__.py 内定义

# 在 _ensure_providers() 中注册到 tool_registry
configs.append(
    _ProviderConfig(
        tool_name="mytool",
        serve_args=["acp", "serve"],
        availability_checker=_mytool_checker,
        model_style="dynamic",
        help_blob_loader=_mytool_help_loader,
    )
)
# result["mytool"] = MyToolACPProvider(cfg)
# tool_registry.register(p, is_default=cfg.is_default)
```

> **说明**：`_ensure_providers()` 采用惰性初始化，只在首次访问 `get_providers()` / `normalize_acp_model_name()` 时构建 checker 与配置，避免增加冷启动开销。

### 2. 在后端 SSOT 注册元数据（`src/mode/backends.py`）

`src/mode/backends.py` 是全仓编程后端的唯一元数据权威（`BackendSpec` / `ALL_BACKEND_SPECS`，同时以 `ALL_BACKEND_METAS` 导出，并经 `src/card/backend_meta.py` 重导出为 `BackendMeta`、`ALL_BACKEND_METAS`、`VISIBLE_BACKEND_METAS`、`BACKEND_META_BY_KEY`、`get_backend_meta`、`resolve_project_programming_mode`）。

新增后端只需在 `src/mode/backends.py` 完成两处声明：

1. 扩展 `InteractionMode` 枚举（例如 `MYTOOL = "mytool"`）。
2. 在 `ALL_BACKEND_SPECS` 中新增一条 `BackendSpec(...)`：

```python
BackendSpec(
    mode_key="mytool",
    interaction_mode=InteractionMode.MYTOOL,
    cli_name="mytool",
    display_name="MyTool",
    emoji="🔧",
    mode_emoji="🔧",
    template_color="blue",
    visible_in_tools_list=True,
    transport="acp",
    uses_claude_cli=False,
    session_timeout_attr="coco_session_timeout",
    is_coco=False,
    legacy_exit_mode="none",
    nl_enter_keywords=("进入mytool模式", "mytool模式", "进入mytool", "使用mytool"),
    supports_info_command=True,
)
```

#### 从 SSOT 自动派生的全链路能力

完成 `ALL_BACKEND_SPECS` 声明后，以下模块会自动同步，**无需再手写 `enter_<tool>_mode` / `is_<tool>_mode` 方法，也无需修改 `src/feishu/ws_client.py` 的路由或意图分发代码**：

- **`InteractionMode` 与 `ModeManager`**（`src/mode/manager.py`）：自动生成 `PROGRAMMING_MODES`、`PROGRAMMING_MODE_VALUES`、`_MODE_DISPLAY_NAMES`，并通过 `_register_programming_mode_accessors` 自动绑定 `ModeManager.enter_<mode_key>_mode()` 与 `ModeManager.is_<mode_key>_mode()`。
- **`ProductCatalog`**（`src/feishu/product_catalog.py`）：自动生成 `/<cli_name>`、`/enter_<mode_key>` 别名与 `/<mode_key>_info` 命令条目，并根据 `visible_in_tools_list` 设置 `slash_discoverable` 与 `HIDDEN_SLASH_COMMANDS`。
- **`SystemHandler` 工具/信息命令映射**（`src/feishu/handlers/system.py`）：自动派生 `_DIRECT_TOOL_COMMAND_TO_MODE`、`_INFO_COMMAND_MODES` 与可拦截控制命令集合。
- **`IntentRecognizer`**（`src/agent/intent_recognizer.py`）：自动派生 `_build_backend_intent_map()`、`_build_backend_exact_commands()`、`_ENTER_TOOL_RULES`、`_INFO_COMMANDS`、`_FALLBACK_INTENT_MAP` 与 `_TOOL_INTENT_MAP`（仍需在 `IntentType` 中补充 `ENTER_<MODE>` 与 `<MODE>_MESSAGE` 枚举值）。
- **`ProjectContext._MODE_ATTRS`**（`src/project/context.py`）：自动生成各模式的活跃标记与会话快照属性映射；同步在 `src/project/unified_context.py` 的 `ContextSourceMode` 与 `src/feishu/handlers/base.py` 的 `mode_to_context_source()` 补充对应枚举映射。
- **`acp.helper._TOOLS`**（`src/acp/helper.py`）：自动派生 `_TOOLS` 与 `_HIDDEN_TOOLS`（供 `/tools`、`/acp` 等发现与过滤）。
- **`HandlerContext` 管理器同步**（`src/feishu/handler_context.py`）：`__post_init__` 按 `ALL_BACKEND_METAS` 双向同步 `managers[mode_key]` 与 `<mode_key>_manager` 属性，所有处理器统一通过 `self.ctx.get_manager(...)` 解析。
- **`FeishuWSClient._initialize()`**（`src/feishu/ws_client.py`）：按 `ALL_BACKEND_METAS` 循环创建每个后端的 `ACPSessionManager`，并通过 `PROGRAMMING_HANDLER_TYPES` 统一注册并实例化所有编程模式 Handler；`src/feishu/action_registry.py`、`src/feishu/dispatcher.py`、`src/card/themes.py`、`src/card/programming_adapter.py`、`src/utils/engine_identity.py`、`src/engine_base.py`、`src/spec_engine/{manager,review}.py` 与 `src/main.py` 均直接读取该 SSOT。

> **UI 文案提示**：若 `visible_in_tools_list=True`，需在 `src/card/ui_text.py` 中补充 `system_acp_tool_desc_<mode_key>` 说明文案；否则一致性格栅测试会自动报错。

### 3. 绑定模式 Handler（`src/feishu/handlers/programming.py`）

在 `src/feishu/handlers/programming.py` 中，使用 `@_bind_programming_backend(InteractionMode.<X>)` 装饰器绑定子类即可：

```python
@_bind_programming_backend(InteractionMode.MYTOOL)
class MyToolModeHandler(ProgrammingModeHandler):
    pass
```

该装饰器会自动从 `BACKEND_SPEC_BY_MODE[interaction_mode]` 注入 `mode_name`、`mode_emoji`、`interaction_mode`、`mode_key`、`context_source`、`thinking_text`（以及 `is_coco` / `_uses_claude_cli` 标记），并将类自动写入 `PROGRAMMING_HANDLER_TYPES[spec.mode_key]`。

基类 `ProgrammingModeHandler._get_session_manager()` 已统一通过 `self.ctx.get_manager(self.mode_key)` 获取会话管理器（并兼容 `ctx.managers` 与 `<mode_key>_manager` 测试替身），子类无需重复实现 `_get_session_manager()`、`_is_in_this_mode()` 或手动模式切换方法。

### 4. 验证注册一致性与测试

完成接入后，必须通过跨层 SSOT 一致性门禁与对应集成测试：

1. **SSOT 跨层同步门禁**：运行 `tests/test_backend_sync_and_actions.py`（验证 `ALL_BACKEND_METAS`、`ModeManager`、`PROGRAMMING_HANDLER_TYPES`、`ProductCatalog`、`SystemHandler`、`IntentRecognizer`、`ProjectContext._MODE_ATTRS`、`acp.helper._TOOLS`、`get_providers()`、`HandlerContext.get_manager()` 等全量对齐）。
2. **Provider 探测与协议测试**：参考 `tests/test_acp_tool_availability.py` 与 `tests/test_acp_client.py`。
3. **后端集成测试**：参考 `tests/test_agy_acp_integration.py`、`tests/test_dsh_acp_integration.py`、`tests/test_grok_acp_integration.py`。

```bash
uv run python -m pytest tests/test_backend_sync_and_actions.py tests/test_acp_tool_availability.py tests/test_acp_client.py -q
```

## 关键设计原则

### ACP 与 CLI 隔离

- **ACP 模式**：`transport="acp"`，直接通过 ACP 协议与工具通信，使用 `SyncACPSession`。
- **CLI 模式**：`transport="cli"`，由专用 CLI 会话后端启动，不伪装成 ACP Provider。
- **强约束**：只有声明支持 ACP 协议的 Provider 才能启动 ACP server。

### 性能优化

- **`hot_tools`**：在 `src/acp/provider.py` 中，`ToolRegistry.get_serve_command()` 对 `hot_tools` 集合中的工具启用异步探活与乐观启动（避免阻塞首轮解析）。
- **可用性预热**：`ToolRegistry.preheat_async()` 支持对核心工具列表进行后台可用性预热。
- **模型目录预热**：`src/main.py` 在启用 `acp_model_preheat_on_startup` 时，直接按 `VISIBLE_BACKEND_METAS` 自动派生预热目标列表。

## 示例与参考

- **SSOT 注册表**：`src/mode/backends.py`、`src/card/backend_meta.py`
- **Provider 配置与特化**：`src/acp/providers/__init__.py`（`GenericACPProvider`、`CodexACPProvider`、`TraexACPProvider`、`GrokACPProvider`）
- **Handler 绑定**：`src/feishu/handlers/programming.py`（`@_bind_programming_backend`）
- **专项接入文档**：[agy_acp_integration.md](agy_acp_integration.md)、[dsh_acp_integration.md](dsh_acp_integration.md)、[grok_acp_integration.md](grok_acp_integration.md)
