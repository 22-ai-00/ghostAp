"""Default constants for the Workflow Engine."""

from __future__ import annotations

# --- Timeouts ---
# NOTE: These are the default-value SSOT / import-time fallbacks. The
# authoritative runtime values are read from Settings (workflow_* fields),
# which allow .env overrides. Keep the numbers here aligned with the Settings
# defaults so any code that still reads the constant directly gets the same
# (more permissive) default.
AGENT_CALL_TIMEOUT_S: int = 600  # Per agent() call timeout (seconds); 0 in Settings disables the per-agent deadline (unlimited)
AGENT_IDLE_TIMEOUT_S: int = 120  # Adaptive idle timeout: kill only after N seconds of no ACP events
SCRIPT_GEN_TIMEOUT_S: int = 180  # AI workflow script generation timeout
WORKFLOW_TOTAL_TIMEOUT_S: int = 3600  # Total workflow execution timeout (60 min); 0 in Settings disables the total deadline (unlimited)
WORKFLOW_TIMEOUT_HEADROOM_S: int = 5  # Reserved seconds before total deadline
SESSION_CREATE_TIMEOUT_S: int = 120

# Finite backstop applied when a per-agent / total timeout is configured as 0
# (unlimited). A blocking future.result() must never wait *forever* — an
# orphaned ACP subprocess would hang the workflow with no way to recover except
# a manual /stop_wf. This backstop is intentionally huge (30 days) so it never
# curtails a legitimately long-running task, while still guaranteeing the call
# eventually returns. Real bounding in unlimited mode comes from the user's
# stop button and the MAX_TOTAL_AGENTS fuse, not this value.
AGENT_UNLIMITED_BACKSTOP_S: int = 30 * 24 * 3600  # 30 days

# --- Concurrency ---
DEFAULT_MAX_CONCURRENT: int = 10  # Default parallel agent slots
HARD_MAX_CONCURRENT: int = 16  # Absolute ceiling regardless of config
MAX_TOTAL_AGENTS: int = 200  # Max agent() calls per workflow run (safety fuse)
MAX_WORKFLOW_AGENT_POOL_SIZE: int = 8  # User-confirmed bindings per Workflow

# --- Nesting ---

# --- Tool descriptions (DEPRECATED — use tool_registry.get_available_tools()) ---
# Kept as import-time fallback; runtime code should use the registry.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "coco": "全栈编程·支持 subagent",
    "aiden": "代码审查·架构设计",
    "codex": "OpenAI 自主编程",
    "claude": "Anthropic 深度推理",
    "traex": "高并发推理·轻量任务",
    "gemini": "Google 多模态推理",
}

# --- Journal ---
JOURNAL_DIR: str = ".ghostap/workflow_journal"
DEFAULT_CACHE_MAX_ENTRIES: int = 100  # Hard cap for in-memory LRU cache size

# --- Schema retry ---
SCHEMA_RETRY_MAX: int = 2  # Max retries when schema validation fails

# --- General retry ---
MAX_RETRIES: int = 3  # Max retries for transient agent call failures
RETRY_BACKOFF_BASE_S: float = 1.0  # Base delay for exponential backoff (seconds)

# --- Queue ---
MAX_QUEUE_SIZE: int = 10_000  # Max pending messages in bridge queue

# --- Runtime ---
RUNTIME_JS_PATH: str = "src/workflow_engine/runtime/runtime.js"
NODE_MIN_VERSION: tuple[int, ...] = (20, 0, 0)
# Number of most-recent Node stderr lines the bridge keeps in a ring buffer.
# Surfaced in the "process exited/closed stdout unexpectedly" diagnostic so a
# crash's dying words (e.g. a V8 FATAL/OOM line) are not lost to the DEBUG-only
# stderr drain. Bounded so a chatty runtime cannot grow memory without limit.
STDERR_TAIL_MAX_LINES: int = 50

# --- Progress ---
PROGRESS_DEBOUNCE_S: float = 2.0  # Max 1 card update per N seconds
# Heartbeat interval for re-rendering the progress card while a long agent()
# call is in flight. Without this, a single multi-minute agent call produces no
# card updates between start and finish, so the card looks "stuck". The
# heartbeat re-renders the running snapshot so the elapsed-time counter keeps
# advancing and the user can see the workflow is still alive and working.
PROGRESS_HEARTBEAT_S: float = 10.0

# --- Script generation ---
# Default agent type used for AI script generation (can be overridden per workflow)
DEFAULT_SCRIPT_GEN_AGENT_TYPE: str = "coco"
# Keep backward compatibility
SCRIPT_GEN_AGENT_TYPE: str = DEFAULT_SCRIPT_GEN_AGENT_TYPE

# Automatic orchestrator default
DEFAULT_ORCHESTRATOR_AGENT: str = "traex"

# --- Engine state filenames ---
STATE_FILENAME: str = ".workflow_engine_state.json"
