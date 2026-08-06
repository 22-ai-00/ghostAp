# GhostAp Slock Mode 改进参考指南：借鉴 @slock-ai/daemon 模式

---

## 1. 执行摘要 (Executive Summary)

GhostAp Slock Engine 与 @slock-ai/daemon 代表了两种不同的 AI Agent 管理哲学：GhostAp 是一个**任务编排器**，擅长多 Agent 协作、智能路由和分层记忆；daemon 是一个**运行时进程管理器**，擅长多运行时适配、会话持久化和事件标准化。经过全面对比分析，GhostAp 在底层运行时管理层面存在代际差距——缺乏统一的 Driver 接口抽象、事件归一化系统、会话恢复机制、遥测能力和限流启动队列。本指南提供了 10 个可借鉴的设计模式，按 P0/P1/P2/P3 四级优先级组织为分阶段实施路线图，目标是在不破坏 GhostAp 现有编排优势的前提下，将其底层执行层提升到生产级可靠性水平。核心改造方向可概括为：**从"管理任务"进化为"管理进程+会话"**。

---

## 2. 架构对比全景 (Architecture Comparison)

### 2.1 总体架构定位

```
┌─────────────────────────────────────────────────────────────────┐
│                    GhostAp Slock Engine                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │TaskRouter│ │MemoryMgr │ │Collabor. │ │  Mouthpiece      │   │
│  │(路由决策) │ │(三层记忆) │ │(DAG编排)  │ │  (消息出口)      │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │
│       │             │            │                 │             │
│  ┌────▼─────────────▼────────────▼─────────────────▼─────────┐  │
│  │              BoundedExecutor (线程池)                       │  │
│  │              ACPSession / SyncClaudeCLI                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│  定位：应用层编排器，Agent = 无状态函数调用                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    @slock-ai/daemon                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │DriverRegistry│  │AgentProcess  │  │ EventNormalizer      │  │
│  │(运行时工厂)   │  │Manager(队列) │  │ (归一化层)           │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                  │                     │              │
│  ┌──────▼──────────────────▼─────────────────────▼───────────┐  │
│  │           ChildProcessRuntimeSession                       │  │
│  │     claude | codex | cursor | gemini | kimi | ...          │  │
│  └───────────────────────────────────────────────────────────┘  │
│  定位：运行时进程管理器，Agent = 有状态进程+持久会话               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 九维度对比总表

| 维度 | GhostAp Slock Engine | @slock-ai/daemon | 胜出方 |
|------|---------------------|------------------|--------|
| 架构范式 | Python 单体 3820 行，推模型，多线程 | TypeScript 模块化，混合推拉，单线程事件循环 | daemon |
| 运行时探测 | 三种异构策略（resolve/custom/which） | 统一 `probe()` 接口 + CLI Transport 隔离 | daemon |
| 生命周期管理 | 9 状态显式状态机 + CAS 转换 | 隐式双状态 + 幂等终态守卫 + 双生命周期模型 | daemon 略胜 |
| 通信协议 | ACP JSON-RPC 统一（强前提假设） | 多协议适配（CLI/MCP/SDK）+ 统一上层接口 | daemon |
| 事件系统 | 6 种 ACP 事件，无内部状态感知 | 11 种归一化事件，含遥测/压缩/活性信号 | daemon |
| 会话恢复 | 仅崩溃降级（IN_PROGRESS -> TODO） | 5 种恢复策略 + per-driver 恢复参数 | daemon |
| 任务路由 | 多级智能路由（chitchat/mention/affinity/skill） | 无本地路由（委托服务端） | **GhostAp** |
| 记忆系统 | 三层分级（L1/L2/L3）+ OCC + 技能积累 | 单层 MEMORY.md 自管理 | **GhostAp** |
| 多Agent协作 | DAG 编排 + Chain + Discussion + Escalation | 无内建协作机制 | **GhostAp** |

### 2.3 关键结论

- **GhostAp 的优势**在上层：路由智能、记忆深度、协作编排
- **daemon 的优势**在底层：运行时抽象、进程管理、事件标准化、会话持久化
- **改进方向**：在保持 GhostAp 上层优势不变的前提下，将底层执行层向 daemon 的模式靠拢

---

## 3. 值得借鉴的模式 (Patterns to Borrow)

### 3.1 [P0] 模式一：统一 Driver 接口（Strategy Pattern）

**What - 是什么：**

将每个运行时（claude/codex/gemini/cursor 等）的行为特征封装为一个实现统一接口的 Driver 类。Driver 通过声明式描述符（descriptor）暴露自己的生命周期类型、通信模式、启停策略等，上层调度器通过接口多态分派行为，消除 `isinstance` / `agent_type` 字符串判断。

**Why - 为什么重要：**

- 当前 GhostAp 添加新运行时需修改 `ACPProvider` 注册代码、新增 Session 子类、在 Manager 中添加分支逻辑——改动分散在 3+ 个文件中
- `agent_type` 字符串在代码中散落为隐式耦合点，IDE 无法追踪所有引用
- 不同 Session 类（`SyncACPSession`, `SyncClaudeCLISession`）的实现不同，上层应只依赖统一协议

**How - 怎么做：**

**步骤 1：定义核心 Protocol**

```python
# src/acp/driver.py
from typing import Protocol, runtime_checkable, Literal, Optional, Any
from dataclasses import dataclass

@dataclass
class LifecycleDescriptor:
    kind: Literal["persistent_stream", "turn_based"]
    post_turn: Literal["terminate_process", "close_stdin", "keep_alive"]
    busy_delivery: Literal["direct", "gated", "none"]
    supports_stdin_notification: bool = False

@dataclass
class ProbeResult:
    available: bool
    version: Optional[str] = None
    error: Optional[str] = None

@dataclass
class NormalizedEvent:
    kind: str  # session_init|thinking|text|tool_call|tool_output|turn_end|error|telemetry|...
    data: dict
    timestamp: float

@runtime_checkable
class RuntimeDriver(Protocol):
    @property
    def id(self) -> str: ...
    
    @property
    def lifecycle(self) -> LifecycleDescriptor: ...
    
    def probe(self) -> ProbeResult: ...
    
    def spawn(self, config: "AgentConfig", workspace: str) -> "RuntimeSessionHandle": ...
    
    def normalize_event(self, raw: Any) -> list[NormalizedEvent]: ...
    
    def build_system_prompt(self, config: "AgentConfig", context: dict) -> str: ...
    
    def encode_message(self, text: str, session_id: Optional[str] = None) -> bytes: ...
    
    def get_resume_args(self, session_id: str) -> list[str]: ...
```

**步骤 2：实现具体 Driver（以 Claude 为例）**

```python
# src/acp/drivers/claude_driver.py
class ClaudeDriver:
    id = "claude"
    lifecycle = LifecycleDescriptor(
        kind="persistent_stream",
        post_turn="keep_alive",
        busy_delivery="direct",
        supports_stdin_notification=True,
    )
    
    def probe(self) -> ProbeResult:
        result = shutil.which("claude")
        if not result:
            return ProbeResult(available=False, error="claude CLI not found")
        # version detection...
        return ProbeResult(available=True, version="1.2.3")
    
    def spawn(self, config, workspace):
        # 原 SyncClaudeCLISession.start() 逻辑迁移至此
        ...
    
    def normalize_event(self, raw):
        # 原 CLI stdout 解析逻辑迁移至此
        ...
```

**步骤 3：Driver Registry 替代 Provider 注册**

```python
# src/acp/driver_registry.py
class DriverRegistry:
    _drivers: dict[str, RuntimeDriver] = {}
    
    @classmethod
    def register(cls, driver: RuntimeDriver):
        cls._drivers[driver.id] = driver
    
    @classmethod
    def get(cls, agent_type: str) -> Optional[RuntimeDriver]:
        return cls._drivers.get(agent_type)
    
    @classmethod
    def probe_all(cls) -> dict[str, ProbeResult]:
        return {d.id: d.probe() for d in cls._drivers.values()}
```

**步骤 4：SessionManager 改为 Driver-parametric**

```python
# 改造前
if agent_type == "claude":
    session = SyncClaudeCLISession(...)
elif agent_type == "codex":
    session = SyncACPSession(...)

# 改造后
driver = DriverRegistry.get(agent_type)
session = driver.spawn(config, workspace)
# session 实现统一的 RuntimeSessionHandle 接口
```

**迁移策略：**
1. Phase 1（1 周）：定义接口 + 为现有 3 种 Session 写 Adapter Wrapper
2. Phase 2（2 周）：逐步将 Session 内部逻辑迁移到 Driver 方法中
3. Phase 3（1 周）：移除旧 Provider 层，统一通过 DriverRegistry 管理

---

### 3.2 [P0] 模式二：事件归一化层（Event Normalizer）

**What - 是什么：**

为每种运行时输出格式实现一个有状态的 Normalizer 类，将异构的原始事件流转换为统一的内部事件 schema。上层消费者（进度追踪、观察学习、消息出口）只依赖归一化后的事件类型。

**Why - 为什么重要：**

- 当前各 Session 类型的事件处理路径完全分离，ProgressTracker 等上层模块需要按 session 类型做分支处理
- 缺少 `turn_end`、`telemetry`、`compaction` 等关键生命周期事件，上层无法准确感知 agent 执行阶段
- 无法统一做日志审计和指标采集

**How - 怎么做：**

```python
# src/acp/normalizers/base.py
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
import time

class EventKind(Enum):
    SESSION_INIT = "session_init"
    THINKING = "thinking"
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    TURN_END = "turn_end"
    ERROR = "error"
    COMPACTION_STARTED = "compaction_started"
    COMPACTION_FINISHED = "compaction_finished"
    TELEMETRY = "telemetry"
    INTERNAL_PROGRESS = "internal_progress"

@dataclass
class NormalizedEvent:
    kind: EventKind
    agent_id: str
    session_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseEventNormalizer:
    """有状态的事件归一化基类"""
    
    def __init__(self, agent_id: str):
        self._agent_id = agent_id
        self._session_id: Optional[str] = None
        self._in_tool_call: bool = False
        self._turn_active: bool = False
    
    def feed(self, raw_event: Any) -> list[NormalizedEvent]:
        """子类实现：将原始事件转换为 0..N 个归一化事件"""
        raise NotImplementedError
    
    def _emit(self, kind: EventKind, data: dict = None) -> NormalizedEvent:
        return NormalizedEvent(
            kind=kind,
            agent_id=self._agent_id,
            session_id=self._session_id,
            data=data or {},
        )
```

```python
# src/acp/normalizers/acp_normalizer.py
class ACPEventNormalizer(BaseEventNormalizer):
    """将 ACP SDK 的 PromptResponse 流事件归一化"""
    
    def feed(self, acp_event) -> list[NormalizedEvent]:
        events = []
        if acp_event.type == "agent_message_chunk":
            events.append(self._emit(EventKind.TEXT, {"content": acp_event.content}))
        elif acp_event.type == "agent_thought_chunk":
            events.append(self._emit(EventKind.THINKING, {"content": acp_event.content}))
        elif acp_event.type == "tool_call_start":
            self._in_tool_call = True
            events.append(self._emit(EventKind.TOOL_CALL, {
                "tool": acp_event.tool_name,
                "input": acp_event.input,
                "status": "started"
            }))
        elif acp_event.type == "tool_call_done":
            self._in_tool_call = False
            events.append(self._emit(EventKind.TOOL_OUTPUT, {
                "tool": acp_event.tool_name,
                "output": acp_event.result,
            }))
        # ... 更多映射
        return events
```

```python
# src/acp/normalizers/cli_normalizer.py
class CLIEventNormalizer(BaseEventNormalizer):
    """将 Claude CLI stdout stream-json 归一化"""
    
    def feed(self, line: str) -> list[NormalizedEvent]:
        events = []
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return events
        
        msg_type = obj.get("type", "")
        
        if msg_type == "system" and "session_id" in obj:
            self._session_id = obj["session_id"]
            events.append(self._emit(EventKind.SESSION_INIT, {
                "session_id": self._session_id
            }))
        elif msg_type == "assistant" and obj.get("subtype") == "thinking":
            events.append(self._emit(EventKind.THINKING, {"content": obj["content"]}))
        elif msg_type == "assistant":
            events.append(self._emit(EventKind.TEXT, {"content": obj["content"]}))
        elif msg_type == "result":
            # 提取 token 使用量作为遥测
            if "usage" in obj:
                events.append(self._emit(EventKind.TELEMETRY, {
                    "input_tokens": obj["usage"].get("input_tokens"),
                    "output_tokens": obj["usage"].get("output_tokens"),
                    "cost_usd": obj["usage"].get("cost"),
                }))
            events.append(self._emit(EventKind.TURN_END, {
                "result": obj.get("result", ""),
                "session_id": self._session_id,
            }))
        
        return events
```

**集成方式：**

```python
# 在 Session 的事件处理循环中注入 Normalizer
class UnifiedSessionWrapper:
    def __init__(self, session, normalizer: BaseEventNormalizer):
        self._session = session
        self._normalizer = normalizer
        self._event_listeners: list[Callable] = []
    
    def on_event(self, listener: Callable[[NormalizedEvent], None]):
        self._event_listeners.append(listener)
    
    def _dispatch(self, raw_event):
        for normalized in self._normalizer.feed(raw_event):
            for listener in self._event_listeners:
                listener(normalized)
```

---

### 3.3 [P0] 模式三：幂等终态守卫（Idempotent Terminal State）

**What - 是什么：**

Session 一旦进入 closed 状态，所有后续操作（send/stop/dispose）立即返回 no-op 结果，无论调用多少次、从哪个线程调用。同时在 kill 之前记录 `stop_reason`，使 exit handler 能区分主动停止与运行时崩溃。

**Why - 为什么重要：**

- 多线程环境下，patrol loop、dispatch loop、超时看门狗可能同时尝试关闭同一 session
- `close()` 中的 asyncio 操作如果重复执行会导致 RuntimeError
- 无法区分 "daemon 主动停止 agent" 与 "agent 进程意外崩溃"，影响后续恢复策略选择

**How - 怎么做：**

```python
import threading
from enum import Enum
from typing import Optional
from dataclasses import dataclass

class StopReason(Enum):
    EXPLICIT_CLOSE = "explicit_close"      # 正常关闭
    USER_CANCEL = "user_cancel"            # 用户取消任务
    TIMEOUT = "timeout"                     # 超时看门狗触发
    RUNTIME_CRASH = "runtime_crash"        # 运行时进程意外退出
    DEACTIVATION = "deactivation"          # 引擎关闭
    ERROR = "error"                         # 不可恢复错误

@dataclass
class CloseResult:
    already_closed: bool
    stop_reason: Optional[StopReason] = None

class IdempotentSessionMixin:
    """混入类，为任何 Session 实现提供幂等终态保护"""
    
    def __init__(self):
        self._closed = False
        self._close_lock = threading.Lock()
        self._stop_reason: Optional[StopReason] = None
    
    @property
    def is_closed(self) -> bool:
        return self._closed
    
    @property
    def stop_reason(self) -> Optional[StopReason]:
        return self._stop_reason
    
    def _try_close(self, reason: StopReason) -> CloseResult:
        """尝试进入终态。返回是否成功（首次关闭）。线程安全。"""
        with self._close_lock:
            if self._closed:
                return CloseResult(already_closed=True, stop_reason=self._stop_reason)
            self._closed = True
            self._stop_reason = reason
            return CloseResult(already_closed=False, stop_reason=reason)
    
    def _guard_closed(self, operation: str) -> bool:
        """在操作前检查是否已关闭。返回 True 表示应 early return。"""
        if self._closed:
            logger.debug(f"Operation '{operation}' skipped: session already closed "
                        f"(reason={self._stop_reason})")
            return True
        return False


# 应用到 SyncACPSession
class SyncACPSession(IdempotentSessionMixin):
    def __init__(self, ...):
        IdempotentSessionMixin.__init__(self)
        # 原有初始化...
    
    def send_prompt(self, text: str, ...) -> PromptResult:
        if self._guard_closed("send_prompt"):
            return PromptResult(text="", error="session_closed", 
                              stop_reason=self._stop_reason)
        # 原有逻辑...
    
    def close(self, reason: StopReason = StopReason.EXPLICIT_CLOSE):
        result = self._try_close(reason)
        if result.already_closed:
            return  # 幂等：不做任何事
        # 实际清理逻辑（只执行一次）...
        self._terminate_process()
        self._cleanup_resources()
    
    def cancel(self):
        # 先设置 reason，再执行 close
        self.close(reason=StopReason.USER_CANCEL)
```

**在 Engine 层的应用：**

```python
# engine.py - deactivate 改造
async def deactivate(self):
    self._run_state = RunState.STOPPING
    
    # 先标记所有 session 为 closing（防止其他循环尝试使用）
    for agent_id, session in self._agent_sessions.items():
        session._try_close(StopReason.DEACTIVATION)
    
    # 再并行执行实际清理
    cleanup_futures = []
    for session in self._agent_sessions.values():
        cleanup_futures.append(
            self._executor.submit(session._cleanup_resources)
        )
    
    # 带超时等待清理完成
    for future in cleanup_futures:
        try:
            future.result(timeout=30)  # 30s graceful timeout
        except TimeoutError:
            logger.warning("Session cleanup timed out, forcing...")
```

---

### 3.4 [P1] 模式四：限流并发启动队列（Rate-Limited Start Queue）

**What - 是什么：**

在 agent 执行线程池之前增加一个启动队列层，控制同时启动的 agent 数量和启动间隔，防止"惊群效应"压垮系统资源。

**Why - 为什么重要：**

- 当前 6 个角色 bootstrap 时会同时发起 6 个子进程，CPU/内存/网络瞬时峰值可能导致部分启动失败
- 崩溃恢复场景下多个 agent 同时重启，竞争加剧
- 无法区分"正在启动"和"正在运行"的 agent，重复启动难以防护

**How - 怎么做：**

```python
# src/acp/start_queue.py
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from typing import Callable, Any
from dataclasses import dataclass
from enum import Enum

class AgentStartState(Enum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"

@dataclass
class StartRequest:
    agent_id: str
    start_fn: Callable[[], Any]
    future: Future
    enqueued_at: float

class AgentStartQueue:
    """限流并发启动队列，防止 agent 启动风暴"""
    
    def __init__(
        self,
        max_concurrent_starts: int = 2,
        start_interval_ms: int = 500,
        on_start_complete: Callable[[str, bool], None] = None,
    ):
        self._max_concurrent = max_concurrent_starts
        self._interval_ms = start_interval_ms
        self._on_complete = on_start_complete
        
        self._lock = threading.Lock()
        self._queued: OrderedDict[str, StartRequest] = OrderedDict()
        self._starting: dict[str, StartRequest] = {}
        self._last_start_time: float = 0
        self._pump_timer: Optional[threading.Timer] = None
        self._shutdown = False
    
    def enqueue(self, agent_id: str, start_fn: Callable[[], Any]) -> Future:
        """入队一个 agent 启动请求。返回 Future 代表启动结果。"""
        with self._lock:
            # 去重：已在队列/启动中则直接返回现有 future
            if agent_id in self._queued:
                return self._queued[agent_id].future
            if agent_id in self._starting:
                return self._starting[agent_id].future
            
            future = Future()
            request = StartRequest(
                agent_id=agent_id,
                start_fn=start_fn,
                future=future,
                enqueued_at=time.time(),
            )
            self._queued[agent_id] = request
        
        self._schedule_pump()
        return future
    
    def _schedule_pump(self):
        """调度下一次泵送检查"""
        with self._lock:
            if self._shutdown:
                return
            if self._pump_timer and self._pump_timer.is_alive():
                return
            
            delay = self._calculate_delay()
            self._pump_timer = threading.Timer(delay, self._pump)
            self._pump_timer.daemon = True
            self._pump_timer.start()
    
    def _calculate_delay(self) -> float:
        elapsed_ms = (time.time() - self._last_start_time) * 1000
        remaining_ms = max(0, self._interval_ms - elapsed_ms)
        return remaining_ms / 1000
    
    def _pump(self):
        """从队列中取出下一个可启动的 agent 并启动"""
        with self._lock:
            if self._shutdown:
                return
            if len(self._starting) >= self._max_concurrent:
                return  # 并发上限，等待
            if not self._queued:
                return  # 队列空
            
            # 取出队首
            agent_id, request = self._queued.popitem(last=False)
            self._starting[agent_id] = request
            self._last_start_time = time.time()
        
        # 在新线程中执行启动
        threading.Thread(
            target=self._execute_start,
            args=(request,),
            daemon=True,
            name=f"agent-start-{agent_id}",
        ).start()
        
        # 如果队列中还有更多，调度下次泵送
        if self._queued:
            self._schedule_pump()
    
    def _execute_start(self, request: StartRequest):
        """执行实际启动逻辑"""
        success = False
        try:
            result = request.start_fn()
            request.future.set_result(result)
            success = True
        except Exception as e:
            request.future.set_exception(e)
        finally:
            with self._lock:
                self._starting.pop(request.agent_id, None)
            if self._on_complete:
                self._on_complete(request.agent_id, success)
            # 启动完成，可能允许下一个启动
            self._schedule_pump()
    
    def cancel(self, agent_id: str) -> bool:
        """取消排队中的启动请求"""
        with self._lock:
            if agent_id in self._queued:
                request = self._queued.pop(agent_id)
                request.future.cancel()
                return True
        return False
    
    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "queued": len(self._queued),
                "starting": len(self._starting),
                "queued_ids": list(self._queued.keys()),
                "starting_ids": list(self._starting.keys()),
            }
    
    def shutdown(self):
        with self._lock:
            self._shutdown = True
            if self._pump_timer:
                self._pump_timer.cancel()
            # 取消所有排队请求
            for request in self._queued.values():
                request.future.cancel()
            self._queued.clear()
```

**集成到 Engine：**

```python
# engine.py
class SlockEngine:
    def __init__(self, ...):
        ...
        self._start_queue = AgentStartQueue(
            max_concurrent_starts=config.get("max_concurrent_starts", 2),
            start_interval_ms=config.get("start_interval_ms", 800),
            on_start_complete=self._on_agent_started,
        )
    
    def _spawn_agent(self, agent_id: str, config: AgentConfig):
        """替代原来的直接 executor.submit"""
        future = self._start_queue.enqueue(
            agent_id=agent_id,
            start_fn=lambda: self._do_spawn(agent_id, config),
        )
        return future
```

---

### 3.5 [P1] 模式五：会话恢复与冷启动策略（Session Resume）

**What - 是什么：**

为每个 agent 维护持久化的 session ID 和执行上下文。崩溃恢复时，根据场景选择不同的恢复策略（带上下文续接 vs 带离线消息摘要 vs 全新冷启动），而非简单地将任务降级重做。

**Why - 为什么重要：**

- 当前 IN_PROGRESS -> TODO 降级意味着 agent 完全丢失已执行的上下文（可能已完成 70% 的工作）
- 长时间任务（如大规模重构）被中断后从头重做，浪费 token 和时间
- 用户体验差：agent 对之前的对话没有记忆

**How - 怎么做：**

```python
# src/acp/session_recovery.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import json
import os

class RecoveryScenario(Enum):
    COLD_START = "cold_start"              # 全新会话
    RESUME_WITH_CONTEXT = "resume_context"  # 恢复 + 注入上次执行上下文
    RESUME_WITH_UPDATES = "resume_updates"  # 恢复 + 离线期间的新消息摘要
    RESUME_EMPTY = "resume_empty"           # 恢复但无新信息
    RESTART_WITH_PARTIAL = "restart_partial" # 新会话但注入上次部分成果

@dataclass
class SessionSnapshot:
    session_id: str
    agent_id: str
    task_id: Optional[str]
    last_turn_result: Optional[str]  # 截断的最后输出
    progress_summary: Optional[str]  # agent 自报的进度摘要
    timestamp: float
    runtime_type: str

@dataclass
class RecoveryDecision:
    scenario: RecoveryScenario
    session_id: Optional[str]  # None 表示新建
    prompt: str                # 恢复 prompt
    cli_args: list[str]        # 传给运行时的额外参数

class SessionRecoveryManager:
    """管理会话快照的持久化和恢复策略决策"""
    
    def __init__(self, storage_dir: str):
        self._storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_snapshot(self, snapshot: SessionSnapshot):
        """在 turn 完成后保存快照"""
        path = os.path.join(self._storage_dir, f"{snapshot.agent_id}.json")
        with open(path, "w") as f:
            json.dump({
                "session_id": snapshot.session_id,
                "agent_id": snapshot.agent_id,
                "task_id": snapshot.task_id,
                "last_turn_result": snapshot.last_turn_result,
                "progress_summary": snapshot.progress_summary,
                "timestamp": snapshot.timestamp,
                "runtime_type": snapshot.runtime_type,
            }, f)
    
    def load_snapshot(self, agent_id: str) -> Optional[SessionSnapshot]:
        """加载上次会话快照"""
        path = os.path.join(self._storage_dir, f"{agent_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return SessionSnapshot(**data)
    
    def decide_recovery(
        self,
        agent_id: str,
        driver: "RuntimeDriver",
        offline_messages: list[str] = None,
        task_still_valid: bool = True,
    ) -> RecoveryDecision:
        """根据场景决定恢复策略"""
        snapshot = self.load_snapshot(agent_id)
        
        # 场景 1：无历史快照 -> 冷启动
        if snapshot is None:
            return RecoveryDecision(
                scenario=RecoveryScenario.COLD_START,
                session_id=None,
                prompt=self._build_cold_start_prompt(agent_id),
                cli_args=[],
            )
        
        # 场景 2：有快照但任务已失效 -> 新会话
        if not task_still_valid:
            return RecoveryDecision(
                scenario=RecoveryScenario.COLD_START,
                session_id=None,
                prompt=self._build_cold_start_prompt(agent_id),
                cli_args=[],
            )
        
        # 场景 3：有快照 + 有离线消息 -> 恢复 + 消息摘要
        if offline_messages:
            summary = self._summarize_messages(offline_messages)
            return RecoveryDecision(
                scenario=RecoveryScenario.RESUME_WITH_UPDATES,
                session_id=snapshot.session_id,
                prompt=f"你离线期间收到了以下消息：\n{summary}\n\n"
                       f"请根据这些信息继续你之前的工作。",
                cli_args=driver.get_resume_args(snapshot.session_id),
            )
        
        # 场景 4：有快照但运行时不支持 resume -> 新会话 + 注入部分成果
        if driver.lifecycle.kind == "turn_based":
            return RecoveryDecision(
                scenario=RecoveryScenario.RESTART_WITH_PARTIAL,
                session_id=None,
                prompt=f"你之前在处理一个任务，以下是你上次的进度：\n"
                       f"{snapshot.progress_summary or snapshot.last_turn_result}\n\n"
                       f"请从上次停止的地方继续。",
                cli_args=[],
            )
        
        # 场景 5：有快照 + 运行时支持 resume + 无新消息 -> 空恢复
        return RecoveryDecision(
            scenario=RecoveryScenario.RESUME_EMPTY,
            session_id=snapshot.session_id,
            prompt="",  # 不发送新 prompt，仅恢复会话
            cli_args=driver.get_resume_args(snapshot.session_id),
        )
```

**在 Engine 中集成：**

```python
# 在 crash recovery 流程中使用
def _recover_agent(self, agent_id: str, task: Task):
    driver = DriverRegistry.get(self._agent_configs[agent_id].type)
    
    decision = self._recovery_manager.decide_recovery(
        agent_id=agent_id,
        driver=driver,
        offline_messages=self._get_offline_messages(agent_id),
        task_still_valid=(task.status != TaskStatus.CANCELLED),
    )
    
    if decision.scenario == RecoveryScenario.COLD_START:
        # 传统路径：重新 spawn + 重新分配任务
        self._spawn_agent(agent_id, ...)
    else:
        # 恢复路径：带 resume 参数启动
        self._spawn_agent_with_resume(agent_id, decision)
```

---

### 3.6 [P1] 模式六：Busy 消息投递策略（Busy Delivery Mode）

**What - 是什么：**

当 agent 正在执行 turn（处于 THINKING/RUNNING/CHECKING 状态）时，定义新消息如何投递给它。三种策略：
- `direct`：直接写入 stdin（中断当前上下文）
- `gated`：缓存，等当前 turn 结束后投递
- `none`：丢弃或路由给其他 agent

**Why - 为什么重要：**

- 当前 GhostAp 在 agent 忙碌时没有明确的消息处理策略
- 用户发送的后续消息可能丢失或被延迟处理但无反馈
- 不同运行时对中断的容忍度不同（Claude 支持 stdin notification，Codex 不支持）

**How - 怎么做：**

```python
# src/acp/busy_delivery.py
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import threading
import time

class DeliveryMode(Enum):
    DIRECT = "direct"   # 直接写入 stdin（运行时支持中断）
    GATED = "gated"     # 缓存到 turn 结束后投递
    NONE = "none"       # 不投递（路由到其他 agent 或丢弃）

@dataclass
class PendingMessage:
    content: str
    sender: str
    timestamp: float = field(default_factory=time.time)
    priority: int = 0

class BusyDeliveryBuffer:
    """管理 agent 忙碌时的消息缓冲"""
    
    def __init__(self):
        self._buffers: dict[str, list[PendingMessage]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def enqueue(self, agent_id: str, message: PendingMessage):
        with self._lock:
            self._buffers[agent_id].append(message)
    
    def drain(self, agent_id: str) -> list[PendingMessage]:
        """取出并清空某 agent 的所有缓冲消息"""
        with self._lock:
            messages = self._buffers.pop(agent_id, [])
        return sorted(messages, key=lambda m: m.timestamp)
    
    def peek_count(self, agent_id: str) -> int:
        with self._lock:
            return len(self._buffers.get(agent_id, []))


# 在消息分发路径中使用
class MessageDispatcher:
    def dispatch(self, agent_id: str, message: str, sender: str):
        driver = DriverRegistry.get(self._get_agent_type(agent_id))
        session = self._get_session(agent_id)
        
        if not session.is_busy:
            # agent 空闲，直接投递
            session.send_prompt(message)
            return
        
        # agent 忙碌，根据 driver 声明的策略处理
        mode = driver.lifecycle.busy_delivery
        
        if mode == DeliveryMode.DIRECT:
            # 运行时支持 stdin notification
            session.send_notification(message)
        elif mode == DeliveryMode.GATED:
            # 缓存，等 turn 结束后投递
            self._buffer.enqueue(agent_id, PendingMessage(
                content=message, sender=sender
            ))
        elif mode == DeliveryMode.NONE:
            # 尝试路由给其他 agent
            alternative = self._router.find_alternative(agent_id, message)
            if alternative:
                self.dispatch(alternative, message, sender)
            else:
                # 无可用 agent，通知发送者
                self._notify_sender(sender, "所有相关 agent 当前忙碌，消息已排队")
                self._buffer.enqueue(agent_id, PendingMessage(
                    content=message, sender=sender
                ))
    
    def on_turn_complete(self, agent_id: str):
        """Turn 完成后投递缓冲消息"""
        pending = self._buffer.drain(agent_id)
        if pending:
            combined = self._combine_messages(pending)
            session = self._get_session(agent_id)
            session.send_prompt(combined)
```

---

### 3.7 [P1] 模式七：遥测采集（Telemetry Collection）

**What - 是什么：**

在事件归一化层采集 token 用量、执行耗时、费用估算、错误率等指标，通过结构化遥测事件上报，为成本控制和性能优化提供数据基础。

**Why - 为什么重要：**

- 当前完全无法追踪每个 agent 消耗了多少 token、产生了多少费用
- 无法识别哪些 agent 效率低下（高 token 消耗但低任务完成率）
- 无法做预算控制和告警

**How - 怎么做：**

```python
# src/telemetry/collector.py
from dataclasses import dataclass, field
from typing import Optional
import time
import threading
import json

@dataclass
class TurnMetrics:
    agent_id: str
    session_id: Optional[str]
    task_id: Optional[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    tool_calls: int = 0
    errors: int = 0
    turn_number: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentCumulativeMetrics:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_turns: int = 0
    total_tool_calls: int = 0
    total_errors: int = 0
    tasks_completed: int = 0
    avg_turn_duration_ms: float = 0.0

class TelemetryCollector:
    """采集和聚合 agent 运行指标"""
    
    def __init__(self, flush_interval_s: int = 60, budget_usd: Optional[float] = None):
        self._lock = threading.Lock()
        self._turn_history: list[TurnMetrics] = []
        self._cumulative: dict[str, AgentCumulativeMetrics] = {}
        self._budget_usd = budget_usd
        self._budget_callbacks: list[callable] = []
    
    def record_turn(self, metrics: TurnMetrics):
        """记录一个 turn 的指标"""
        with self._lock:
            self._turn_history.append(metrics)
            
            # 更新累计指标
            cum = self._cumulative.setdefault(
                metrics.agent_id, AgentCumulativeMetrics()
            )
            cum.total_input_tokens += metrics.input_tokens
            cum.total_output_tokens += metrics.output_tokens
            cum.total_cost_usd += metrics.cost_usd
            cum.total_turns += 1
            cum.total_tool_calls += metrics.tool_calls
            cum.total_errors += metrics.errors
            cum.avg_turn_duration_ms = (
                (cum.avg_turn_duration_ms * (cum.total_turns - 1) + metrics.duration_ms)
                / cum.total_turns
            )
        
        # 预算检查
        if self._budget_usd is not None:
            total_cost = sum(c.total_cost_usd for c in self._cumulative.values())
            if total_cost >= self._budget_usd * 0.8:
                for cb in self._budget_callbacks:
                    cb(total_cost, self._budget_usd)
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentCumulativeMetrics]:
        with self._lock:
            return self._cumulative.get(agent_id)
    
    def get_total_cost(self) -> float:
        with self._lock:
            return sum(c.total_cost_usd for c in self._cumulative.values())
    
    def on_budget_warning(self, callback):
        self._budget_callbacks.append(callback)
    
    def export_report(self) -> dict:
        """导出完整报告"""
        with self._lock:
            return {
                "total_cost_usd": self.get_total_cost(),
                "agents": {
                    aid: {
                        "input_tokens": m.total_input_tokens,
                        "output_tokens": m.total_output_tokens,
                        "cost_usd": m.total_cost_usd,
                        "turns": m.total_turns,
                        "tasks_completed": m.tasks_completed,
                        "avg_turn_ms": m.avg_turn_duration_ms,
                        "error_rate": m.total_errors / max(m.total_turns, 1),
                    }
                    for aid, m in self._cumulative.items()
                }
            }
```

**与事件归一化层集成：**

```python
# 在 UnifiedSessionWrapper 中自动采集
class TelemetryEventListener:
    def __init__(self, collector: TelemetryCollector):
        self._collector = collector
        self._current_turn: dict[str, TurnMetrics] = {}
    
    def on_event(self, event: NormalizedEvent):
        agent_id = event.agent_id
        
        if event.kind == EventKind.SESSION_INIT:
            # 新 turn 开始
            self._current_turn[agent_id] = TurnMetrics(
                agent_id=agent_id,
                session_id=event.session_id,
            )
        elif event.kind == EventKind.TOOL_CALL:
            if agent_id in self._current_turn:
                self._current_turn[agent_id].tool_calls += 1
        elif event.kind == EventKind.ERROR:
            if agent_id in self._current_turn:
                self._current_turn[agent_id].errors += 1
        elif event.kind == EventKind.TELEMETRY:
            if agent_id in self._current_turn:
                turn = self._current_turn[agent_id]
                turn.input_tokens = event.data.get("input_tokens", 0)
                turn.output_tokens = event.data.get("output_tokens", 0)
                turn.cost_usd = event.data.get("cost_usd", 0.0)
        elif event.kind == EventKind.TURN_END:
            if agent_id in self._current_turn:
                turn = self._current_turn.pop(agent_id)
                turn.duration_ms = int((time.time() - turn.timestamp) * 1000)
                self._collector.record_turn(turn)
```

---

### 3.8 [P2] 模式八：Context Window 压缩感知

**What - 是什么：**

监测 agent 的上下文窗口使用率，在接近限制时感知运行时自动压缩行为（compaction），或主动触发会话重启/摘要注入，防止上下文溢出导致的静默质量退化。

**Why - 为什么重要：**

- 长时间运行的 persistent agent 上下文窗口会逐渐填满
- Claude 等运行时在压缩时会丢失细节，但上层系统不知道这发生了
- 可能导致 agent 遗忘早期指令或重要上下文

**How - 怎么做：**

```python
# src/acp/context_monitor.py
from dataclasses import dataclass
from typing import Optional, Callable
import threading

@dataclass
class ContextWindowState:
    total_tokens: int
    used_tokens: int
    compaction_count: int = 0
    last_compaction_at: Optional[float] = None
    
    @property
    def usage_ratio(self) -> float:
        return self.used_tokens / self.total_tokens if self.total_tokens > 0 else 0

class ContextWindowMonitor:
    """监控 agent 上下文窗口状态"""
    
    HIGH_WATER_MARK = 0.75   # 75% 触发警告
    CRITICAL_MARK = 0.90     # 90% 触发主动干预
    MAX_COMPACTIONS = 3       # 超过此次数建议重启会话
    
    def __init__(self):
        self._states: dict[str, ContextWindowState] = {}
        self._callbacks: dict[str, list[Callable]] = {
            "high_water": [],
            "critical": [],
            "compaction": [],
            "recommend_restart": [],
        }
    
    def update_from_telemetry(self, agent_id: str, input_tokens: int, model: str):
        """从遥测数据更新上下文状态"""
        max_tokens = self._get_model_limit(model)
        state = self._states.setdefault(
            agent_id, 
            ContextWindowState(total_tokens=max_tokens, used_tokens=0)
        )
        state.used_tokens = input_tokens
        
        # 检查水位
        ratio = state.usage_ratio
        if ratio >= self.CRITICAL_MARK:
            self._fire("critical", agent_id, state)
        elif ratio >= self.HIGH_WATER_MARK:
            self._fire("high_water", agent_id, state)
    
    def on_compaction_event(self, agent_id: str):
        """运行时报告了压缩事件"""
        state = self._states.get(agent_id)
        if state:
            state.compaction_count += 1
            state.last_compaction_at = time.time()
            self._fire("compaction", agent_id, state)
            
            if state.compaction_count >= self.MAX_COMPACTIONS:
                self._fire("recommend_restart", agent_id, state)
    
    def on(self, event: str, callback: Callable):
        self._callbacks[event].append(callback)
    
    def _fire(self, event: str, agent_id: str, state: ContextWindowState):
        for cb in self._callbacks.get(event, []):
            cb(agent_id, state)
    
    @staticmethod
    def _get_model_limit(model: str) -> int:
        limits = {
            "claude-sonnet": 200_000,
            "claude-opus": 200_000,
            "gpt-4": 128_000,
            "codex": 128_000,
            "gemini-pro": 1_000_000,
        }
        for key, limit in limits.items():
            if key in model.lower():
                return limit
        return 128_000  # 默认
```

**集成到 Engine：**

```python
# 在 Engine 初始化时设置回调
self._context_monitor = ContextWindowMonitor()

self._context_monitor.on("recommend_restart", self._handle_context_restart)

def _handle_context_restart(self, agent_id: str, state: ContextWindowState):
    """上下文多次压缩，建议重启会话"""
    logger.warning(
        f"Agent {agent_id} context compacted {state.compaction_count} times, "
        f"recommending session restart"
    )
    # 保存当前进度快照
    self._recovery_manager.save_snapshot(...)
    # 优雅重启
    self._restart_agent_session(agent_id, reason="context_exhaustion")
```

---

### 3.9 [P2] 模式九：进程级 Idle/Keep-Alive 管理（Post-Turn Strategy）

**What - 是什么：**

Turn 结束后，根据运行时特性和资源状况选择不同的进程处理策略：
- `keep_alive`：保持进程运行（persistent runtime，保留上下文）
- `close_stdin`：关闭输入流但保留进程（等待可能的后续请求）
- `terminate_process`：终止进程（turn-based runtime，释放资源）

**Why - 为什么重要：**

- 当前所有 agent 线程执行完即结束，无法复用已有的 agent 上下文
- 对于 persistent runtime（如 Claude），每次都重新 spawn 意味着丢失对话历史
- 空闲 agent 进程占用内存但不做任何事，应有超时回收机制

**How - 怎么做：**

```python
# src/acp/idle_manager.py
from enum import Enum
from typing import Optional
import threading
import time

class PostTurnAction(Enum):
    KEEP_ALIVE = "keep_alive"
    CLOSE_STDIN = "close_stdin"  
    TERMINATE = "terminate_process"

class IdleAgentManager:
    """管理空闲 agent 进程的生命周期"""
    
    DEFAULT_IDLE_TIMEOUT_S = 300  # 5 分钟无活动则回收
    
    def __init__(self, idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S):
        self._idle_timeout = idle_timeout_s
        self._idle_since: dict[str, float] = {}  # agent_id -> last_active_time
        self._lock = threading.Lock()
        self._running = False
        self._scan_thread: Optional[threading.Thread] = None
    
    def start(self):
        self._running = True
        self._scan_thread = threading.Thread(
            target=self._idle_scan_loop, daemon=True, name="idle-scan"
        )
        self._scan_thread.start()
    
    def mark_active(self, agent_id: str):
        """标记 agent 为活跃（有新 turn）"""
        with self._lock:
            self._idle_since.pop(agent_id, None)
    
    def mark_idle(self, agent_id: str):
        """标记 agent 为空闲（turn 结束）"""
        with self._lock:
            self._idle_since[agent_id] = time.time()
    
    def _idle_scan_loop(self):
        while self._running:
            time.sleep(30)  # 每 30s 扫描一次
            self._evict_stale()
    
    def _evict_stale(self):
        now = time.time()
        to_evict = []
        with self._lock:
            for agent_id, idle_since in self._idle_since.items():
                if now - idle_since > self._idle_timeout:
                    to_evict.append(agent_id)
        
        for agent_id in to_evict:
            self._on_idle_timeout(agent_id)
    
    def _on_idle_timeout(self, agent_id: str):
        """空闲超时回调 - 由 Engine 注册具体行为"""
        # 保存会话快照后终止
        ...
```

---

### 3.10 [P3] 模式十：Agent 活性探测（Liveness Probe）

**What - 是什么：**

定期检测 agent 进程/线程是否仍然活跃（非死锁/卡死），通过内部进度事件或显式心跳判断。对无响应的 agent 执行超时回收和任务重分配。

**Why - 为什么重要：**

- 当前仅依赖 SLA 超时（默认 300s），但 API 卡死时无法提前发现
- 心跳机制可以区分"agent 在思考"（有 internal_progress）和"agent 已死"（完全无输出）

**How - 怎么做：**

```python
# src/acp/liveness.py
import threading
import time
from typing import Optional, Callable

class LivenessProbe:
    """基于最后活动时间的活性探测"""
    
    # 如果一段时间内没有任何事件（包括 internal_progress），判定为可能卡死
    SUSPECT_THRESHOLD_S = 120   # 2 分钟无事件 -> 可疑
    DEAD_THRESHOLD_S = 300      # 5 分钟无事件 -> 判死
    
    def __init__(self, on_suspect: Callable, on_dead: Callable):
        self._last_activity: dict[str, float] = {}
        self._lock = threading.Lock()
        self._on_suspect = on_suspect
        self._on_dead = on_dead
        self._running = False
    
    def heartbeat(self, agent_id: str):
        """任何事件到达时调用"""
        with self._lock:
            self._last_activity[agent_id] = time.time()
    
    def start_monitoring(self, agent_id: str):
        with self._lock:
            self._last_activity[agent_id] = time.time()
    
    def stop_monitoring(self, agent_id: str):
        with self._lock:
            self._last_activity.pop(agent_id, None)
    
    def check_all(self):
        """定期调用，检查所有被监控的 agent"""
        now = time.time()
        with self._lock:
            items = list(self._last_activity.items())
        
        for agent_id, last_active in items:
            elapsed = now - last_active
            if elapsed >= self.DEAD_THRESHOLD_S:
                self._on_dead(agent_id, elapsed)
            elif elapsed >= self.SUSPECT_THRESHOLD_S:
                self._on_suspect(agent_id, elapsed)
```

**与事件系统集成：**

```python
# 每收到 NormalizedEvent 就更新心跳
def on_normalized_event(self, event: NormalizedEvent):
    self._liveness_probe.heartbeat(event.agent_id)
    # ... 其他处理
```

---

## 4. 差距分析 (Gap Analysis)

### 4.1 完全缺失的能力

| 编号 | 缺失能力 | 影响范围 | 严重度 | 对应借鉴模式 |
|------|----------|---------|--------|-------------|
| G1 | 多运行时驱动架构 | 无法低成本接入 cursor/copilot/kimi 等新运行时 | **严重** | 模式 1 |
| G2 | 事件归一化系统 | 无法统一监控/日志/遥测，各 session 处理分散 | **严重** | 模式 2 |
| G3 | 会话恢复机制 | 崩溃后丢失所有执行上下文，浪费大量 token | **严重** | 模式 5 |
| G4 | 运行时遥测 | 无法追踪成本、无法做预算控制、无法识别低效 agent | **高** | 模式 7 |
| G5 | Context Window 压缩感知 | 长对话质量静默退化无感知 | **高** | 模式 8 |
| G6 | 速率受限启动队列 | Bootstrap 和崩溃恢复时的资源风暴 | **高** | 模式 4 |
| G7 | Busy 消息投递策略 | Agent 忙碌时消息可能丢失或处理不当 | **中** | 模式 6 |
| G8 | 进程级 Idle 管理 | 无法复用 persistent runtime 上下文 | **中** | 模式 9 |
| G9 | Agent 活性探测 | 卡死 agent 无法提前发现，等待 SLA 超时才回收 | **中** | 模式 10 |
| G10 | MCP 工具命名空间 | 多运行时场景下工具名冲突 | **低** | 未列入模式 |

### 4.2 实现更弱的现有模式

| 编号 | 现有模式 | 当前问题 | 改进方向 |
|------|----------|---------|---------|
| W1 | Agent 状态机 | 9 状态但无幂等终态保护，多次 close 可能异常 | 模式 3 |
| W2 | Executor shutdown | `shutdown(wait=False)` 强制中断，可能损坏数据 | 模式 3 + graceful timeout |
| W3 | 全局 RLock | TaskBoardManager 单锁瓶颈 | 分段锁/ConcurrentDict |
| W4 | BoundedExecutor 硬编码 | max_workers=4 无法按需调整 | 可配置 + 模式 4 队列 |
| W5 | 执行结果截断 | 仅保留最后 2000 字符，关键上下文可能丢失 | 结构化结果 + 摘要策略 |

### 4.3 可扩展性天花板

```
当前系统在以下规模下会遇到瓶颈：

Agent 数量:     > 6 个 (BoundedExecutor 硬上限 + GIL 竞争)
并发任务:       > 10 个 (全局 RLock + 线性扫描)
消息吞吐:       > 50 msg/min (单线程 dispatch loop)
持续运行时间:   > 2 小时 (无上下文压缩感知 + 无内存回收)
崩溃恢复:       每次崩溃丢失 100% 执行上下文
```

---

## 5. 实施路线图 (Implementation Roadmap)

### 5.1 总体时间线

```
Phase 0 (基础设施) ─── 第 1-2 周
  ├── 定义 RuntimeDriver Protocol
  ├── 定义 NormalizedEvent schema
  └── 实现 IdempotentSessionMixin

Phase 1 (核心改造) ─── 第 3-6 周
  ├── 实现 3 个 Driver (claude/codex/gemini)
  ├── 实现 3 个 EventNormalizer
  ├── AgentStartQueue 上线
  └── TelemetryCollector 集成

Phase 2 (可靠性提升) ─── 第 7-10 周
  ├── SessionRecoveryManager 实现
  ├── BusyDeliveryBuffer 集成
  ├── ContextWindowMonitor 上线
  └── IdleAgentManager + LivenessProbe

Phase 3 (优化) ─── 第 11-12 周
  ├── 分段锁替换全局 RLock
  ├── Executor graceful shutdown
  └── 端到端集成测试
```

### 5.2 详细行动计划

#### Phase 0：基础设施（P0 - 第 1-2 周）

| 任务 | 交付物 | 依赖 | 验收标准 |
|------|--------|------|---------|
| 定义 RuntimeDriver Protocol | `src/acp/driver.py` | 无 | mypy 类型检查通过 |
| 定义 NormalizedEvent schema | `src/acp/normalizers/base.py` | 无 | 覆盖 11 种事件类型 |
| 实现 IdempotentSessionMixin | `src/acp/session_guard.py` | 无 | 并发 close 测试通过 |
| 为现有 3 种 Session 写 Adapter | `src/acp/drivers/*_adapter.py` | Protocol | 不改变现有行为 |

**里程碑：** 现有系统通过 Adapter 层运行，接口定义完成，无行为变更。

#### Phase 1：核心改造（P0/P1 - 第 3-6 周）

| 任务 | 交付物 | 依赖 | 验收标准 |
|------|--------|------|---------|
| ClaudeDriver 完整实现 | `src/acp/drivers/claude_driver.py` | Protocol | probe + spawn + normalize 全链路 |
| CodexDriver 完整实现 | `src/acp/drivers/codex_driver.py` | Protocol | 同上 |
| GeminiDriver 完整实现 | `src/acp/drivers/gemini_driver.py` | Protocol | 同上 |
| ACPEventNormalizer | `src/acp/normalizers/acp_normalizer.py` | schema | 正确映射所有 ACP 事件 |
| CLIEventNormalizer | `src/acp/normalizers/cli_normalizer.py` | schema | 正确解析 Claude CLI 输出 |
| AgentStartQueue | `src/acp/start_queue.py` | 无 | 并发限制 + 速率限制测试 |
| TelemetryCollector | `src/telemetry/collector.py` | Normalizer | token/cost 统计准确 |
| SessionManager 改造 | `src/acp/session_manager.py` | Drivers | 通过 Driver 实例化 session |

**里程碑：** 新旧系统并行运行，通过 feature flag 切换。所有现有测试通过。

#### Phase 2：可靠性提升（P1/P2 - 第 7-10 周）

| 任务 | 交付物 | 依赖 | 验收标准 |
|------|--------|------|---------|
| SessionRecoveryManager | `src/acp/session_recovery.py` | Drivers | 5 种场景正确恢复 |
| BusyDeliveryBuffer | `src/acp/busy_delivery.py` | Normalizer | 三种模式正确投递 |
| ContextWindowMonitor | `src/acp/context_monitor.py` | Telemetry | 压缩感知 + 自动重启 |
| IdleAgentManager | `src/acp/idle_manager.py` | 无 | 超时回收 + 快照保存 |
| LivenessProbe | `src/acp/liveness.py` | Normalizer | 卡死检测 + 自动恢复 |
| Graceful Shutdown 改造 | engine.py | Guard | 30s 超时等待 + force kill |

**里程碑：** 系统具备完整的故障恢复和资源管理能力。

#### Phase 3：优化（P2/P3 - 第 11-12 周）

| 任务 | 交付物 | 依赖 | 验收标准 |
|------|--------|------|---------|
| TaskBoard 分段锁 | task_board.py | 无 | 并发性能提升 >50% |
| 结构化执行结果 | models.py | Normalizer | 不再简单截断 |
| 端到端集成测试 | tests/e2e/ | 所有模块 | 覆盖所有恢复场景 |
| 文档更新 | docs/ | 所有模块 | 新架构文档完整 |

**里程碑：** 全面切换到新架构，移除旧代码路径。

### 5.3 每个 Phase 的风险缓解

| Phase | 主要风险 | 缓解措施 |
|-------|---------|---------|
| Phase 0 | 接口定义不完整，后续需频繁修改 | 参考 daemon 的 Driver 接口完整列表，预留 extension point |
| Phase 1 | 改造过程中破坏现有功能 | Feature flag 控制新旧路径，AB 测试验证 |
| Phase 2 | 会话恢复逻辑复杂，边界条件多 | 每种恢复场景单独写测试用例，chaos engineering |
| Phase 3 | 分段锁实现复杂度高 | 考虑用 Python concurrent.futures 或 asyncio 替代手写锁 |

---

## 6. 风险与注意事项 (Risks & Caveats)

### 6.1 技术风险

| 风险 | 可能性 | 影响 | 缓解方案 |
|------|--------|------|---------|
| Python GIL 限制多 Driver 并发效果 | 高 | 中 | I/O 密集型操作已自然释放 GIL；计算密集操作考虑 multiprocessing |
| 事件归一化层成为性能瓶颈 | 低 | 高 | Normalizer 保持无分配热路径，必要时用 C extension |
| 会话恢复与运行时版本不兼容 | 中 | 高 | 快照包含运行时版本号，版本变更时退回冷启动 |
| Adapter 层引入额外间接性 | 高 | 低 | Phase 1 完成后立即内联 Adapter，消除间接层 |
| 新旧代码并行期间维护成本翻倍 | 高 | 中 | 严格控制并行期时间（不超过 4 周），定期清理旧代码 |

### 6.2 架构决策注意事项

**不应照搬的 daemon 设计：**

1. **服务端路由委托** — daemon 将路由决策完全交给服务端。GhostAp 的本地智能路由（TaskRouter）是核心竞争力，应保留并增强。

2. **单层记忆模型** — daemon 仅有 per-agent MEMORY.md。GhostAp 的三层记忆架构（L1/L2/L3）和 OCC 并发控制是更成熟的设计，不应简化。

3. **无协作编排** — daemon 没有内建的多 Agent 协作机制。GhostAp 的 CollaborationOrchestrator 和 TaskChainManager 是差异化优势。

4. **隐式状态机** — daemon 用 flags 管理状态而非显式状态机。GhostAp 的 9 状态显式状态机在正确性验证方面更强，但应补充幂等终态守卫（模式 3）作为增强而非替换。

**应保持的 GhostAp 设计优势：**

- 多级任务路由（chitchat 过滤 + @mention + affinity + skill score）
- 三层分级记忆 + OCC 并发控制
- DAG 编排 + Chain 模板 + Discussion 状态机
- Escalation 三级升级机制
- Collaboration Plan 人工审批流程

### 6.3 实施原则

1. **加法优先**：优先通过新增类/模块实现，避免大规模重写现有代码
2. **接口契约先行**：先定义 Protocol/ABC，再实现具体类，确保可测试性
3. **Feature Flag 控制**：所有新路径通过配置开关控制，可随时回退
4. **向后兼容**：Phase 1 期间必须保证所有现有功能不受影响
5. **渐进式迁移**：一个 Driver 一个 Driver 地迁移，不做 big bang

### 6.4 成功指标

| 指标 | 当前基线 | Phase 1 目标 | Phase 3 目标 |
|------|---------|-------------|-------------|
| 新增运行时接入时间 | 2-3 天（改 3+ 文件） | 1 天（仅写 1 个 Driver） | 半天 |
| 崩溃恢复上下文保留率 | 0%（完全重做） | 60%（带摘要恢复） | 90%（会话续接） |
| Token 成本可见性 | 无 | per-agent 粒度 | per-task 粒度 |
| Agent 卡死检测时间 | 300s（SLA 超时） | 120s（活性探测） | 60s |
| 并发启动稳定性 | 偶发失败 | 0 失败（队列化） | 0 失败 |
| 上下文溢出静默失败 | 不可观测 | 可感知+告警 | 自动恢复 |

---

*本文档最后更新：2026-06-05*
*适用范围：GhostAp Slock Engine v1.x -> v2.x 架构升级*
