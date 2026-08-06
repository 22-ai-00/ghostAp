# ACP Provider 接入指南

## 概述

本指南说明如何在 GhostAP 系统中新增一个支持 ACP（Agent Client Protocol）协议的 AI 开发工具。

## 架构概览

系统已提供完整的通用 ACP 协议适配层：

1. **ACPProvider Protocol** - 统一工具接入接口
2. **ToolRegistry** - 工具注册/发现机制
3. **SyncACPSession** - ACP 会话管理
4. **传输边界** - ACP Provider 与 shell CLI 后端保持独立

## 接入步骤

### 1. 创建 Provider 实现

在 `src/acp/providers/` 目录下创建新的 provider 文件，例如 `mytool.py`：

```python
"""MyTool Provider（ACP 模式）"""

from __future__ import annotations
from typing import Optional
from ..provider import ACPProvider


class MyToolProvider(ACPProvider):
    @property
    def name(self) -> str:
        return "mytool"

    def check_availability(self) -> bool:
        """判断工具是否可用且支持 `acp serve`"""
        try:
            from ..sync_adapter import _probe_acp_serve_help
            ok, _rc, _out, _err = _probe_acp_serve_help("mytool")
            return bool(ok)
        except Exception:
            return False

    def get_serve_command(self, model_name: Optional[str] = None) -> tuple[str, list[str]]:
        """生成 ACP Server 启动命令"""
        args = ["acp", "serve"]
        if model_name:
            # 根据工具的 CLI 参数格式添加 model 参数
            args.extend(["--model", model_name])
        return "mytool", args

    def get_fallback_command(self, model_name: Optional[str] = None) -> Optional[tuple[str, list[str]]]:
        """可选：在不可用时提供可执行的兜底命令"""
        return None  # 或返回兜底命令
```

### 2. 注册 Provider

在 `src/acp/providers/__init__.py` 中注册新 provider：

```python
from .mytool import MyToolProvider

# Register standard providers
tool_registry.register(MyToolProvider())
```

同时更新 `__all__`：

```python
__all__ = [
    # ... 现有项 ...
    "MyToolProvider",
]
```

### 3. 添加模式 Handler（如需要）

如果工具需要独立的交互模式，在 `src/feishu/handlers/programming.py` 中添加：

```python
class MyToolModeHandler(ProgrammingModeHandler):
    mode_name = "MyTool"
    mode_emoji = "🔧"
    is_coco = False
    context_source = ContextSourceMode.MYTOOL
    thinking_text = "🔧 MyTool 正在思考..."

    def _get_session_manager(self):
        return self.ctx.mytool_manager

    def _is_in_this_mode(self, chat_id):
        return self.mode_manager.is_mytool_mode(chat_id)

    # ... 实现其他抽象方法
```

### 4. 更新 Intent Recognizer

在 `src/agent/intent_recognizer.py` 中添加：

1. 扩展 `IntentType` enum
2. 添加精确命令匹配
3. 添加关键词匹配

### 5. 更新 Mode Manager

在 `src/mode/manager.py` 中添加：

1. 扩展 `InteractionMode` enum
2. 添加 `enter_mytool_mode()` 方法
3. 添加 `is_mytool_mode()` 方法

### 6. 更新 WebSocket Client

在 `src/feishu/ws_client.py` 中：

1. 添加 handler 实例化
2. 更新 `_FORWARDING_MAP`
3. 添加 intent 处理逻辑

## 关键设计原则

### ACP 与 CLI 隔离

- **ACP 模式**：直接通过 ACP 协议与工具通信，使用 `SyncACPSession`
- **CLI 模式**：由专用 CLI 会话后端启动，不伪装成 ACP Provider
- **强约束**：只有声明 ACP 能力的 Provider 才能启动 ACP server

### 性能优化

- **hot_tools**：在 `src/acp/provider.py` 中将新工具添加到 `hot_tools` 集合，享受异步探活和乐观启动
- **preheat_async**：在 `preheat_async` 默认预热列表中添加新工具

### 测试要求

新增工具时，请确保添加：

1. Provider 层单元测试（参考 `tests/test_acp_provider_extensions.py`）
2. Intent 识别测试
3. 模式切换集成测试

## 示例

参考现有实现：

- **Coco**: `src/acp/providers/coco.py`
- **Claude**: `src/acp/providers/claude.py`
- **Aiden**: `src/acp/providers/aiden.py`
- **Codex**: `src/acp/providers/codex.py`
