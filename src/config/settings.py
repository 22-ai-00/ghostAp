"""Settings — main application configuration model backed by pydantic-settings."""

import base64
import binascii
import logging as _logging
import math
import os
import re
import shlex
import warnings
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.autonomous.config import AutonomousDeploymentMode

from .card import CardSessionConfig
from .spec import SpecReviewConfig

# ---------------------------------------------------------------------------
# Slock discussion constants (copied from src/slock_engine/discussion_manager.py)
# Keep in sync with discussion_manager.py when updating.
# Using static copies here to avoid circular imports and ensure Settings can
# be imported independently without triggering slock_engine imports.
# ---------------------------------------------------------------------------

# Signals that indicate discussion convergence (agents agreeing)
_DEFAULT_CONVERGENCE_SIGNALS: frozenset[str] = frozenset({
    "AGREE",
    "LGTM",
    "同意",
    "认可",
    "没问题",
    "looks good",
    "sounds good",
    "no further suggestions",
})

# Markers that indicate uncertainty in agent output (triggers discussion)
_DEFAULT_UNCERTAINTY_MARKERS: frozenset[str] = frozenset({
    "不确定",
    "需要确认",
    "需要讨论",
    "needs review",
    "需要审查",
    "not sure",
    "i'm not sure",
    "uncertain",
    "maybe",
    "可能",
    "也许",
})


def _emit_slock_prefix_deprecation_warning() -> None:
    warnings.warn(
        "SLOCK_TEAM_NAME_PREFIX is deprecated, use SLOCK_TEAM_NAME_SUFFIX instead",
        DeprecationWarning,
        stacklevel=3,
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_id: str = ""
    app_secret: str = ""

    # Default ACP tool for SMART mode (e.g., "coco", "claude", "aiden", "codex", "gemini", "traex")
    # When set, unmatched messages in SMART mode are forwarded to this tool.
    # When empty, all unmatched messages are treated as shell commands.
    default_acp_tool: str = ""

    sandbox_timeout: int = 30
    sandbox_max_output_length: int = 4000
    sandbox_command_blacklist: str = ""
    sandbox_use_whitelist: bool = False
    sandbox_command_whitelist: str = ""
    ingress_access_mode: Literal[
        "enforced", "shadow", "legacy_allow_all"
    ] = "enforced"
    admin_bootstrap_scope: Literal["any_chat", "p2p_only"] = "p2p_only"
    shell_access_mode: Literal[
        "disabled", "admin_dm", "allowlisted", "isolated", "trusted_local"
    ] = "disabled"
    shell_trusted_local_ack: bool = False
    employee_department_enabled: bool = False
    employee_group_context_retention_days: int = Field(
        default=30,
        ge=1,
        le=3650,
    )

    # 允许 set_working_dir 切换到的根目录列表
    # 校验时会通过 realpath 解析 symlink，所以 ~/work 能匹配 /data00/home/x/work
    project_allowed_roots: list[str] = [
        os.path.expanduser("~/workspaces"),
        os.path.expanduser("~/work"),
    ]

    coco_execution_timeout: int = Field(default=7200, gt=0)
    coco_session_timeout: int = 86400
    coco_max_output_length: int = 30000

    claude_execution_timeout: int = Field(default=7200, gt=0)
    claude_session_timeout: int = 86400
    claude_max_output_length: int = 30000
    programming_finalization_reserve_s: int = Field(
        default=600,
        ge=0,
        le=3600,
        description=(
            "Seconds reserved from ordinary programming execution time for a "
            "separate bounded finalization prompt; 0 disables the reserve."
        ),
    )

    # ACP session history directory (empty = default ~/.ghostap/acp_history)
    acp_history_dir: str = ""

    # ACP agent process startup timeout (seconds)
    acp_startup_timeout: int = 20

    # ACP agent startup retries (1 means no retry)
    acp_startup_retries: int = 2

    # ACP health check timeout (seconds)
    acp_healthcheck_timeout: float = 2.0

    # ACP model list probe timeout (seconds). Much larger than healthcheck:
    # cold-spawning `coco acp serve` + initialize + new_session round-trip is
    # highly variable and routinely takes 5-12s on first use (observed range
    # 4-12s). A tight 6s window times out often, and falling back to the static
    # DEFAULT_MODELS hides the real model list (GPT-5.x, GLM-5, Kimi, openrouter
    # pools, Gemini previews, …), so we give the probe a generous window before
    # degrading. The startup preheat keeps the 5min cache warm so /coco normally
    # never pays this cost interactively.
    acp_model_probe_timeout: float = 15.0

    # Warm up the coco ACP model list in the background at startup so the
    # interactive /coco model picker reads a fresh 5min-cached list instead of
    # paying the cold-spawn probe cost (and risking a timeout → stale defaults).
    acp_model_preheat_on_startup: bool = True

    # ACP permission auto-approve (True = agent actions auto-approved, False = denied by default)
    acp_permission_auto_approve: bool = True

    # Auto-update agent CLI when ACP server mode is not supported
    acp_auto_update: bool = True
    # Timeout for agent CLI auto-update subprocess (seconds)
    acp_auto_update_timeout: int = 120

    # Engine eval prompt timeout (seconds) — used by Spec engine
    engine_eval_prompt_timeout: int = 60

    # Engine auxiliary prompt timeout (seconds) — used by disposable ACP
    # sub-sessions for non-critical tasks such as Spec criteria decomposition.
    # Keep this aligned with SyncACPSession's default prompt timeout to avoid
    # cold-start/model-selection latency causing noisy 60s timeout errors before
    # the main Spec cycle even begins.
    engine_aux_prompt_timeout: int = 600

    # ACP stdio stream buffer limit (bytes). Default asyncio limit is 64KB which
    # is too small for large agent responses (code generation, file contents).
    # Set to 0 to use the asyncio default (64KB). 10MB should be generous enough.
    acp_stream_buffer_limit: int = 10 * 1024 * 1024

    acp_keepalive_interval: int = 300

    acp_session_idle_healthcheck_s: float = 120.0

    # Maximum characters for file content in ACP read/write operations
    acp_max_file_chars: int = 200_000

    # ------------------------------------------------------------------
    # ACP startup diagnostics (redaction + truncation)
    # ------------------------------------------------------------------
    # Safety-first: redact sensitive values from diagnostics logs.
    acp_diagnostics_redact_enabled: bool = True
    acp_diagnostics_redact_replacement: str = "***REDACTED***"
    # Regex patterns applied to args/stdout_snippet/stderr_snippet/spec strings.
    # NOTE: Keep patterns conservative to avoid excessive false positives.
    acp_diagnostics_redact_patterns: list[str] = [
        r"(?i)authorization\s*:\s*[^\s]+",
        r"(?i)bearer\s+[^\s]+",
        r"sk-[A-Za-z0-9]{10,}",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)api[_-]?key\s*[:=]\s*[^\s]+",
        r"(?i)secret\s*[:=]\s*[^\s]+",
        r"(?i)token\s*[:=]\s*[^\s]+",
    ]
    # Unified truncation limits for diagnostics output.
    # - args_limit: approximated length of joined args (best-effort)
    # - snippet_limit: stdout/stderr snippet length
    # - total_limit: final formatted JSON line length
    acp_diagnostics_args_limit: int = 600
    acp_diagnostics_snippet_limit: int = 240
    acp_diagnostics_total_limit: int = 2000

    # Claude CLI backend: skip Claude's built-in permission checks.
    # GhostAP has its own sandbox safety layer, so this is usually safe.
    claude_cli_skip_permissions: bool = True

    # ACP agent command overrides (optional)
    # Example:
    #   COCO_ACP_CMD=coco
    #   COCO_ACP_ARGS="acp serve"
    coco_acp_cmd: str = ""
    coco_acp_args: str = ""
    claude_acp_cmd: str = ""
    claude_acp_args: str = ""

    # ------------------------------------------------------------------
    # Workflow Engine
    # ------------------------------------------------------------------

    # Sub-workflow budget ratio: fraction of parent budget allocated to each sub-workflow.
    # Prevents a single sub-workflow from consuming the entire parent budget.
    # Range: 5% to 50% (enforced via Field validation).
    workflow_subflow_budget_ratio: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        description="Fraction of parent budget allocated to each sub-workflow (0.05-0.5)",
    )

    # Subagent / workflow encouragement hint: when True (default), every
    # agent() prompt template and every script-generation prompt template
    # includes a trailing paragraph that encourages the model to delegate to
    # subagents and invoke sub-workflows.  Set to False to suppress the hint
    # (e.g. for short one-shot calls where the extra verbosity is not useful).
    workflow_subagent_hint_enabled: bool = True

    # Workflow timeout knobs (runtime-overridable via .env). Defaults are kept
    # aligned with the import-time fallbacks in workflow_engine/constants.py.
    # Raise these for complex, long-running /wf tasks that hit the deadline.
    workflow_total_timeout_s: int = Field(
        default=3600,
        ge=0,
        description="Total /wf workflow execution timeout (seconds); SSOT for the run deadline. Set 0 to disable the total deadline entirely (unlimited) — the workflow then runs until it finishes or the user stops it. Per-agent timeout and the MAX_TOTAL_AGENTS fuse still apply.",
    )
    workflow_agent_call_timeout_s: int = Field(
        default=600,
        ge=0,
        description="Per agent() call timeout inside a workflow (seconds). Set 0 to disable the per-agent deadline (unlimited) — a single agent() call then runs until it finishes or the user stops the workflow. The MAX_TOTAL_AGENTS fuse and (if set) the total-workflow deadline still apply. When >0 this value is the authoritative floor: it overrides any smaller per-call timeout baked into the generated script so long-running coding tasks are not killed prematurely.",
    )
    workflow_agent_idle_timeout_s: int = Field(
        default=120,
        ge=0,
        description="Adaptive idle timeout for agent() calls (seconds). When >0, the per-agent timeout becomes activity-based: as long as ACP events (tool calls, text output) keep arriving, the agent stays alive. Only after this many seconds of complete silence is the agent killed. The hard cap (workflow_agent_call_timeout_s) still applies as an absolute maximum. Set 0 to disable adaptive timeout and use fixed timeout only.",
    )
    workflow_script_gen_timeout_s: int = Field(
        default=180,
        ge=10,
        description="AI workflow script generation timeout (seconds).",
    )
    workflow_session_create_timeout_s: int = Field(
        default=120,
        ge=10,
        description="Agent session creation timeout inside a workflow (seconds).",
    )


    # Spec Engine settings
    spec_max_cycles: int = 1000
    # Hard upper bound for long-range spec cycles (configurable via env).
    # Engine will clamp spec_max_cycles to this limit.
    spec_max_cycles_limit: int = 5000
    spec_execution_timeout: int = 7200
    spec_convergence_window: int = 2
    spec_min_cycles: int = 2
    spec_review_enabled: bool = True
    spec_review_strategy: Literal["adaptive_roles", "multi_perspective", "none"] = "adaptive_roles"
    spec_review_timeout: int = 240
    spec_review_role_timeout_multipliers: dict[str, float] = {"architect": 1.5}
    spec_review_max_parallel: int = 3
    spec_review_dynamic_roles_enabled: bool = True
    spec_review_dynamic_roles_max: int = 3
    spec_review_total_roles_max: int = 8
    spec_review_pass_streak_required: int = 2

    # Spec Engine review failure circuit breaker
    # - enabled: master switch
    # - max_consecutive: open circuit after N consecutive review failures
    # - cooldown_cycles: keep circuit open for next K cycles (skip review)
    spec_review_failure_circuit_enabled: bool = True
    spec_review_failure_max_consecutive: int = 4
    spec_review_failure_cooldown_cycles: int = 2
    spec_review_failure_max_cooldown_cycles: int = 12
    spec_review_min_timeout: int = 60
    spec_review_hard_floor: int = 20

    # Spec Engine review in-cycle auto-retry (max_attempts=0 disables retry)
    spec_review_retry_max_delay: int = 30
    spec_review_retry_max_attempts: int = 2
    spec_review_retry_base_delay: float = 8.0
    spec_review_retry_decay_factor: float = 1.5

    # 审查会话启动超时（秒），独立于全局 acp_startup_timeout
    spec_review_startup_timeout: int = 30

    # 审查解析失败时的默认判定 ("fail" = 视为未通过, "pass" = 视为通过)
    spec_review_parse_failure_default: Literal["pass", "fail"] = "fail"

    # ---- 完成度把控（Completion Control）----
    spec_objective_verify_enabled: bool = True
    spec_objective_verify_timeout: int = 300
    spec_completion_control_active_verify: bool = True
    spec_completion_gate_enabled: bool = True

    @property
    def spec_review(self) -> "SpecReviewConfig":
        """Structured view of spec review / retry / circuit-breaker settings."""
        return SpecReviewConfig(
            enabled=self.spec_review_enabled,
            timeout=self.spec_review_timeout,
            max_parallel=self.spec_review_max_parallel,
            min_timeout=self.spec_review_min_timeout,
            hard_floor=self.spec_review_hard_floor,
            retry_max_delay=self.spec_review_retry_max_delay,
            retry_max_attempts=self.spec_review_retry_max_attempts,
            retry_base_delay=self.spec_review_retry_base_delay,
            retry_decay_factor=self.spec_review_retry_decay_factor,
            failure_circuit_enabled=self.spec_review_failure_circuit_enabled,
            failure_max_consecutive=self.spec_review_failure_max_consecutive,
            failure_cooldown_cycles=self.spec_review_failure_cooldown_cycles,
            failure_max_cooldown_cycles=self.spec_review_failure_max_cooldown_cycles,
            parse_failure_default=self.spec_review_parse_failure_default,
            strategy=self.spec_review_strategy,
            dynamic_roles_enabled=self.spec_review_dynamic_roles_enabled,
            dynamic_roles_max=self.spec_review_dynamic_roles_max,
            total_roles_max=self.spec_review_total_roles_max,
            pass_streak_required=self.spec_review_pass_streak_required,
        )

    # Streaming card collapsible panels (tool calls / thoughts folded by default)
    # Engine card collapsible panels (Deep/Spec: structured content with collapsible panels)
    engine_collapsible_enabled: bool = True

    # Streaming card auto-continuation (create new card when content exceeds threshold)
    # Card session / delivery / UI configuration (nested model)
    card: CardSessionConfig = CardSessionConfig()

    # Review metrics exporter
    # - "logger" (default): output via logging.info (original behaviour)
    # - "jsonl": append JSON Lines to review_metrics_jsonl_path
    review_metrics_exporter_type: str = "logger"
    review_metrics_jsonl_path: str = "review_metrics.jsonl"

    # Sliding window dynamic circuit breaker (used by Spec engine)
    # - window_size: number of recent review outcomes to track (min 3)
    # - success_rate_threshold: open circuit if success_rate < threshold
    review_circuit_window_size: int = 10
    review_circuit_success_rate_threshold: float = 0.3

    # Review circuit-breaker lint fallback (run local lint when circuit is open)
    review_circuit_lint_fallback_enabled: bool = True
    review_circuit_lint_timeout: int = 10

    # Spec long-range persistence / monitoring
    # Empty = mirror project absolute paths under ~/.cache/ghostAp.
    spec_cache_root: str = ""
    spec_state_filename: str = ".spec_engine_state.json"
    spec_artifacts_dirname: str = ".spec_engine"
    # Keep in-memory phase outputs bounded for 5k+ cycles
    spec_cycle_output_max_chars: int = 4000
    spec_cycle_tasks_max: int = 50
    # Persisted artifact bounds / retention (avoid 5k cycles generating huge disk usage)
    spec_phase_output_persist_max_chars: int = 20000
    spec_cycle_artifact_retention: int = 50
    # Whether to persist phase raw outputs (spec/plan/tasks/build/review) to disk.
    # Metrics/state/spec files are still persisted for long-range monitoring/resume.
    spec_persist_phase_artifacts: bool = True
    # Post-cycle self-questioning (problem discovery) + spec generation
    spec_discovery_enabled: bool = True
    spec_discovery_max_questions: int = 5
    spec_discovery_force_nonempty: bool = True
    spec_generated_specs_per_cycle: int = 3
    # Discovery 门控（防空转）
    spec_discovery_gate_on_satisfied: bool = True  # AC 全满足后关闭 discovery
    spec_discovery_max_pending: int = 5  # backlog 达上限时跳过 discovery
    spec_discovery_cooldown_cycles: int = 3  # 无进展时每 N 轮才触发一次
    # Termination 增强
    spec_backlog_stuck_window: int = 0  # backlog_stuck 检测窗口 (0=禁用，要求全部消化)
    spec_success_ignore_backlog: bool = False  # success 判定时要求 backlog 清零
    # Persistence cadence
    spec_persist_every_phase: bool = True
    spec_allow_resume_from_disk: bool = True
    # Continuation policy
    # - infinite_mode: never stop due to convergence/early-stop; only stop on success/user stop/max_cycles
    spec_infinite_mode: bool = False
    spec_disable_convergence: bool = False
    spec_disable_early_stop: bool = False
    spec_rebuild_session_between_cycles: bool = True
    # State file compaction (avoid O(n^2) rewrite cost for 5k cycles)
    spec_state_cycles_tail: int = 50
    spec_state_work_items_tail: int = 200
    spec_state_metrics_tail: int = 200

    # History / retention
    spec_history_log_filename: str = "history.jsonl"
    spec_max_retries: int = 3
    # Max consecutive cycle failures before aborting (prevents infinite empty loops).
    spec_max_consecutive_failures: int = 3
    spec_model_switch_enabled: bool = True
    spec_generated_specs_retention: int = 1000
    # Override hint: when set (non-empty), mask BUILD phase errors to "Internal error"
    spec_failed_task_id_override: str = ""

    streaming_enabled: bool = True

    # Feishu WebSocket reconnect delay (seconds) when underlying client exits unexpectedly
    feishu_ws_reconnect_delay_s: float = 5.0

    # Feishu WebSocket watchdog interval (seconds)
    feishu_ws_watchdog_interval: float = 60.0

    # ------------------------------------------------------------------
    # Feishu WebSocket client runtime parameters
    # ------------------------------------------------------------------
    # 消息过期时间（秒），超时的历史消息不再处理
    message_expire_seconds: int = 30
    # 消息去重缓存 TTL（秒）
    message_cache_ttl: int = 300
    # 消息去重缓存最大容量
    message_cache_max_size: int = 1000
    # 消息去重缓存清理间隔（秒）
    message_cache_cleanup_interval: int = 60
    # 系统命令并发数
    system_command_concurrency: int = 10
    # Spec 引擎任务限流容量
    spec_rate_limit_capacity: int = 100
    # Spec 引擎任务限流填充速率（tokens/sec）
    spec_rate_limit_fill_rate: float = 50.0
    # Spec 引擎任务熔断阈值（连续失败次数）
    spec_circuit_breaker_threshold: int = 10
    # Spec 引擎任务熔断恢复超时（秒）
    spec_circuit_breaker_recovery: float = 5.0

    # Streaming flow control (Adaptive interval)
    streaming_adaptive_interval_base: float = 0.3  # Base interval (seconds) for low rate
    streaming_adaptive_interval_max: float = 2.0  # Max interval (seconds) for high rate
    streaming_adaptive_rate_low: float = 20.0  # Low rate threshold (chars/sec)
    streaming_adaptive_rate_high: float = 150.0  # High rate threshold (chars/sec)

    # ------------------------------------------------------------------
    # IM API / Deep Streaming Control
    # ------------------------------------------------------------------
    # Maximum retries for IM API patch operations (default: 3)
    im_api_max_retries: int = 3

    # Deep engine streaming update throttling
    # - interval: minimum seconds between updates (unless forced)
    # - min_chars: minimum new characters accumulated before updating (unless forced/interval passed)
    deep_stream_interval: float = 1.5
    deep_stream_min_chars: int = 350

    # Deep engine memory monitoring (percentage)
    deep_memory_threshold: float = 80.0

    # Rate limiting handling (auto-pause and retry on API throttling)
    rate_limit_retry_enabled: bool = True
    rate_limit_max_wait: int = 300  # Max seconds to wait for rate limit cooldown
    rate_limit_base_wait: int = 30  # Default wait if no retry-after header
    rate_limit_max_retries: int = 5  # Max consecutive rate limit retries

    # Engine timeout warning threshold (seconds) for long-running tasks
    engine_timeout_warning_seconds: int = 600

    # ------------------------------------------------------------------
    # Model failure self-healing (send_prompt-time)
    # ------------------------------------------------------------------
    # need compaction / loop detected 防抖与 failover 参数
    model_failure_compaction_enabled: bool = True
    model_failure_compaction_loop_window_s: float = 180.0
    model_failure_compaction_loop_max: int = 2
    # failover mapping (comma/space separated: "from:to")
    # default: gpt-5.2 -> gpt-5.1
    model_failure_failover_map: str = "gpt-5.2:gpt-5.1"

    # Task scheduler (thread-based) settings
    task_scheduler_max_concurrent: int = 20
    task_scheduler_per_key_concurrency: int = 1
    # Cross-process restart gate. Empty means the checkout's stable locator
    # chooses a private gate under the checkout parent's registry; an override
    # must be an absolute dedicated directory.
    restart_gate_dir: str = Field(
        default="",
        validation_alias=AliasChoices(
            "restart_gate_dir",
            "GHOSTAP_RESTART_GATE_DIR",
        ),
    )
    restart_gate_timeout: float = Field(
        default=7200.0,
        gt=0,
        validation_alias=AliasChoices(
            "restart_gate_timeout",
            "GHOSTAP_RESTART_GATE_TIMEOUT",
        ),
    )

    @field_validator("restart_gate_dir")
    @classmethod
    def _validate_restart_gate_dir(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("restart gate directory must be absolute")
        normalized = Path(os.path.abspath(path))
        if (
            normalized
            in {
                Path("/"),
                Path("/tmp"),
                Path("/var/tmp"),
                Path("/private/tmp"),
                Path("/dev/shm"),
            }
            or len(normalized.parts) <= 2
            or normalized == Path.home()
        ):
            raise ValueError(
                "restart gate directory must be a dedicated private directory"
            )
        return str(normalized)

    @field_validator("restart_gate_timeout")
    @classmethod
    def _validate_restart_gate_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("restart gate timeout must be finite and > 0")
        return value

    # 消息回复模式配置
    # - direct: 直接回复（消息显示在被回复消息下方）
    # - thread: 话题回复（使用 reply_in_thread=True，消息会显示在独立话题区域，更整洁）
    #
    # smart_reply_mode: 智能模式下的回复方式（默认 direct，群内直接引用消息回复）
    # default_reply_mode: 其他模式（Coco/Claude/Shell/Deep等）的回复方式（默认 thread，话题回复更整洁）
    smart_reply_mode: str = "direct"
    default_reply_mode: str = "thread"

    thread_programming_enabled: bool = True
    thread_context_ttl: int = 86400 * 7

    # ref-note 关联信息开关（默认关闭，调试时可通过 .env 设置 REF_NOTE_ENABLED=true）
    ref_note_enabled: bool = False

    # ------------------------------------------------------------------
    # RepoLockManager — 仓库操作锁
    # ------------------------------------------------------------------
    repo_lock_idle_timeout: int = 300  # 锁空闲超时（秒），超时自动释放（仅 refcount=0 时生效）
    repo_lock_cleanup_interval: int = 60  # 清理线程扫描间隔（秒）
    repo_lock_hard_timeout: int = 3600  # 活跃锁续租超时（秒），无心跳超时后强制回收

    # ChatLockManager — 群锁 TTL
    chat_lock_max_duration: int = 86400  # 群锁最大持续时间（秒，默认 24h），超时自动释放
    chat_lock_cleanup_interval: int = 60  # 群锁清理线程扫描间隔（秒）

    # /lock 撤销窗口时长（秒），用户锁定后可在此窗口内撤销
    lock_undo_window_seconds: int = 300

    # /lock 确认卡片有效期（秒），超时后确认按钮失效
    lock_confirm_timeout: int = 120

    # SandboxExecutor 严格锁模式 — True 时检测到锁冲突 raise LockConflictError，False 仅 warning
    sandbox_strict_lock_mode: bool = False

    # ------------------------------------------------------------------
    # 签名回退兼容窗口 — 升级后旧按钮的 plain SHA-256 签名过渡期
    # ------------------------------------------------------------------
    sig_compat_deploy_date: str = ""  # ISO 格式部署日期，回退窗口起点；空值时以进程启动日期为起点
    sig_compat_window_days: int = 7  # 回退兼容天数，超过后仅接受 HMAC 签名

    # ------------------------------------------------------------------
    # 管理员用户列表（用于群级锁权限判定）
    # Stored as frozenset for O(1) membership checks on hot paths.
    # Declared as str to prevent pydantic-settings from attempting JSON parse
    # on plain comma-separated values; converted to frozenset in model_validator.
    # ------------------------------------------------------------------
    admin_user_ids: str = ""

    # ------------------------------------------------------------------
    # Autonomous Work System
    # Configuration requests a deployment surface; effective write autonomy
    # is derived separately from runtime safety attestations.
    # ------------------------------------------------------------------
    autonomous_deployment_mode: AutonomousDeploymentMode = AutonomousDeploymentMode.OFF
    autonomous_compatibility_mode: Literal["legacy", "shadow_read", "manager_only", "disabled"] = "legacy"
    autonomous_write_enabled: bool = False
    autonomous_state_dir: str = "~/.ghostap/autonomy"
    autonomous_journal_dir: str = "~/.ghostap/autonomy/journal"
    autonomous_journal_hmac_key: SecretStr = SecretStr("")
    autonomous_snapshot_dir: str = "~/.ghostap/autonomy/snapshots"
    autonomous_blob_dir: str = "~/.ghostap/autonomy/blobs"
    autonomous_credential_dir: str = "~/.ghostap/slock/credentials"
    autonomous_credential_keys: SecretStr = SecretStr("")
    autonomous_credential_active_key_id: str = ""
    autonomous_data_keys: SecretStr = SecretStr("")
    autonomous_data_active_key_id: str = ""
    autonomous_data_blob_dir: str = "~/.ghostap/autonomy/data-blobs"
    autonomous_history_timezone: str = "UTC"
    autonomous_history_max_range_days: int = Field(default=31, ge=1, le=366)
    autonomous_history_page_size: int = Field(default=50, ge=1, le=200)
    autonomous_thread_context_max_messages: int = Field(
        default=200,
        ge=1,
        le=10_000,
    )
    autonomous_thread_context_max_chars: int = Field(
        default=400_000,
        ge=1,
        le=5_000_000,
    )
    autonomous_group_context_max_messages: int = Field(
        default=50,
        ge=1,
        le=500,
    )
    autonomous_context_max_tokens: int = Field(
        default=128_000,
        ge=1,
        le=2_000_000,
    )
    autonomous_thread_context_page_size: int = Field(default=50, ge=1, le=50)
    autonomous_group_context_page_size: int = Field(default=20, ge=1, le=50)
    autonomous_context_fetch_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        allow_inf_nan=False,
    )
    autonomous_context_retry_base_seconds: float = Field(
        default=1.0,
        gt=0,
        le=300,
        allow_inf_nan=False,
    )
    autonomous_context_retry_max_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        allow_inf_nan=False,
    )
    autonomous_team_step_timeout_seconds: float = Field(
        default=600.0,
        gt=0,
        le=3600,
        allow_inf_nan=False,
    )
    # Persistent coordination is the normal employee-team runtime. The legacy
    # path remains an explicit rollback mode until real-tenant soak is signed.
    autonomous_team_runtime_mode: Literal["legacy_pipeline", "coordinator"] = (
        "coordinator"
    )
    autonomous_team_coordinator_tool: str = "coco"
    autonomous_team_coordinator_model: str = ""
    autonomous_team_coordinator_profile: str = ""
    autonomous_team_coordinator_effort: str = ""
    autonomous_fire_grace_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        allow_inf_nan=False,
    )
    autonomous_context_max_pages: int = Field(default=200, ge=1, le=10_000)
    autonomous_employee_ingress_ack_timeout_seconds: float = Field(
        default=1.5,
        gt=0,
        lt=3.0,
        allow_inf_nan=False,
    )
    autonomous_employee_ingress_max_payload_bytes: int = Field(
        default=256 * 1024,
        ge=1,
        le=256 * 1024,
    )
    autonomous_employee_ingress_max_attachment_count: int = Field(
        default=10,
        ge=1,
        le=10,
    )
    autonomous_employee_ingress_max_attachment_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1,
        le=20 * 1024 * 1024,
    )
    autonomous_employee_ingress_max_total_attachment_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
        le=50 * 1024 * 1024,
    )
    autonomous_employee_ingress_blob_dir: str = (
        "~/.ghostap/autonomy/ingress-blobs"
    )
    autonomous_employee_outbox_blob_dir: str = (
        "~/.ghostap/autonomy/outbox-blobs"
    )
    autonomous_employee_attachment_staging_dir: str = (
        "~/.ghostap/autonomy/employee-attachments"
    )
    autonomous_employee_system_prompt_token_reserve: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
    )
    # ``actor`` is the normal employee runtime and never falls back to one-shot.
    # ``shadow`` and ``legacy_one_shot`` remain explicit rollback diagnostics.
    autonomous_employee_runtime_mode: Literal[
        "legacy_one_shot", "shadow", "actor"
    ] = "actor"
    autonomous_employee_session_idle_ttl_seconds: float = Field(
        default=900.0,
        gt=0,
        le=86_400,
        allow_inf_nan=False,
    )
    autonomous_employee_traex_auth_home: str = "~/.trae"
    autonomous_employee_queue_per_employee_limit: int = Field(default=8, ge=1, le=10_000)
    autonomous_employee_queue_per_team_limit: int = Field(default=32, ge=1, le=100_000)
    autonomous_employee_queue_global_limit: int = Field(default=128, ge=1, le=1_000_000)
    autonomous_manager_acl: str = ""
    autonomous_anchor_provider: str = ""
    autonomous_anchor_path: str = "~/.ghostap/autonomy/journal.anchor"
    autonomous_sandbox_required: bool = True
    autonomous_worker_sandbox_verified: bool = False
    autonomous_oracle_sandbox_verified: bool = False
    autonomous_memory_enabled: bool = False
    autonomous_vc_enabled: bool = False
    autonomous_goal_queue_limit: int = Field(default=1000, ge=1)
    autonomous_run_queue_limit: int = Field(default=100, ge=1)
    autonomous_visible_employee_limit: int = Field(default=8, ge=0)
    autonomous_employee_auto_activation: bool = True
    autonomous_employee_release_evidence_bundle: str = (
        "~/.ghostap/autonomy/employee-release-evidence.jsonl"
    )
    autonomous_employee_release_checkpoint: str = (
        "~/.ghostap/autonomy/employee-release-attestation.json"
    )
    autonomous_employee_release_public_key: str = ""
    autonomous_employee_release_key_id: str = ""
    autonomous_employee_release_id: str = ""
    autonomous_employee_commit_sha: str = ""
    autonomous_employee_service_instance_id: str = ""
    autonomous_employee_staging_tenant_hash: str = ""
    autonomous_employee_production_tenant_hash: str = ""
    autonomous_employee_release_trust_socket: str = ""
    autonomous_employee_release_trust_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        le=30,
    )
    autonomous_main_bot_audit_dir: str = (
        "~/.ghostap/autonomy/main-bot-send-audit"
    )
    autonomous_main_bot_audit_anchor_path: str = (
        "~/.ghostap/autonomy/main-bot-send-audit.anchor"
    )

    @field_validator(
        "autonomous_thread_context_max_messages",
        "autonomous_thread_context_max_chars",
        "autonomous_group_context_max_messages",
        "autonomous_context_max_tokens",
        "autonomous_thread_context_page_size",
        "autonomous_group_context_page_size",
        "autonomous_context_fetch_timeout_seconds",
        "autonomous_context_retry_base_seconds",
        "autonomous_context_retry_max_seconds",
        "autonomous_team_step_timeout_seconds",
        "autonomous_fire_grace_seconds",
        "autonomous_context_max_pages",
        "autonomous_employee_ingress_ack_timeout_seconds",
        "autonomous_employee_ingress_max_payload_bytes",
        "autonomous_employee_ingress_max_attachment_count",
        "autonomous_employee_ingress_max_attachment_bytes",
        "autonomous_employee_ingress_max_total_attachment_bytes",
        "autonomous_employee_queue_per_employee_limit",
        "autonomous_employee_queue_per_team_limit",
        "autonomous_employee_queue_global_limit",
        "autonomous_employee_system_prompt_token_reserve",
        mode="before",
    )
    @classmethod
    def _reject_boolean_context_settings(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("employee context/ingress numeric settings reject booleans")
        return value

    @field_validator(
        "autonomous_team_coordinator_tool",
        "autonomous_team_coordinator_model",
        "autonomous_team_coordinator_profile",
        "autonomous_team_coordinator_effort",
    )
    @classmethod
    def _validate_team_coordinator_identity(
        cls, value: str, info: ValidationInfo
    ) -> str:
        if value != value.strip():
            raise ValueError("team coordinator settings reject surrounding whitespace")
        if info.field_name == "autonomous_team_coordinator_tool" and not value:
            raise ValueError("team coordinator tool is required")
        return value

    @model_validator(mode="after")
    def _validate_employee_queue_limit_order(self) -> "Settings":
        if not (
            self.autonomous_employee_queue_per_employee_limit
            <= self.autonomous_employee_queue_per_team_limit
            <= self.autonomous_employee_queue_global_limit
        ):
            raise ValueError("employee queue limits require per_employee <= per_team <= global")
        return self

    @model_validator(mode="after")
    def _validate_context_retry_delay_order(self) -> "Settings":
        if (
            self.autonomous_context_retry_base_seconds
            > self.autonomous_context_retry_max_seconds
        ):
            raise ValueError("context retry requires base seconds <= maximum seconds")
        return self

    @field_validator("autonomous_journal_hmac_key", mode="before")
    @classmethod
    def _validate_autonomous_journal_hmac_key(cls, value: object) -> SecretStr:
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if raw in (None, ""):
            return SecretStr("")
        if not isinstance(raw, str):
            raise ValueError("autonomous_journal_hmac_key must be base64 text")
        try:
            decoded = base64.b64decode(raw, altchars=b"-_", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "autonomous_journal_hmac_key must be valid base64"
            ) from exc
        if len(decoded) < 32:
            raise ValueError(
                "autonomous_journal_hmac_key must decode to at least 32 bytes"
            )
        return SecretStr(raw)

    @field_validator("autonomous_anchor_path")
    @classmethod
    def _validate_autonomous_anchor_path(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("autonomous_anchor_path must be a non-empty path")
        return value

    @field_validator("autonomous_anchor_provider")
    @classmethod
    def _validate_autonomous_anchor_provider(cls, value: str) -> str:
        if value == "":
            return value
        if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", value) is None:
            raise ValueError(
                "autonomous_anchor_provider must be a stable lowercase identifier"
            )
        return value

    @field_validator("autonomous_employee_release_trust_timeout_seconds", mode="before")
    @classmethod
    def _reject_boolean_release_trust_timeout(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("release trust timeout rejects booleans")
        return value

    @field_validator(
        "autonomous_employee_release_trust_socket",
        "autonomous_main_bot_audit_dir",
        "autonomous_main_bot_audit_anchor_path",
    )
    @classmethod
    def _validate_employee_release_paths(cls, value: str) -> str:
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("employee release paths must be text without NUL")
        return value

    @field_validator("autonomous_history_timezone")
    @classmethod
    def _validate_autonomous_history_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (TypeError, ZoneInfoNotFoundError) as exc:
            raise ValueError("autonomous_history_timezone must be a valid IANA timezone") from exc
        return value

    # ------------------------------------------------------------------
    # 授权白名单（安全加固 A2）
    # 空 frozenset 表示不限制（允许所有）；非空时仅白名单内的 chat/user 可用。
    # Declared as str for same reason as admin_user_ids.
    # ------------------------------------------------------------------
    allowed_chat_ids: str = ""
    allowed_user_ids: str = ""

    @field_validator("admin_user_ids", "allowed_chat_ids", "allowed_user_ids", mode="before")
    @classmethod
    def _normalize_id_set_input(cls, v):
        """Normalize list/set/frozenset input to comma-separated string."""
        if isinstance(v, (list, tuple, set, frozenset)):
            return ",".join(v)
        return v if v is not None else ""

    @model_validator(mode="after")
    def _coerce_id_set_fields(self) -> "Settings":
        """Convert comma-separated id-set fields to frozenset for O(1) lookup."""
        for field_name in ("admin_user_ids", "allowed_chat_ids", "allowed_user_ids"):
            raw = getattr(self, field_name)
            if not raw or not isinstance(raw, str):
                parsed = frozenset()
            else:
                parsed = frozenset(s.strip() for s in raw.split(",") if s.strip())
            object.__setattr__(self, field_name, parsed)
        return self

    # ------------------------------------------------------------------
    # 项目 chat 隔离 — allowed_chat_ids 上限
    # ------------------------------------------------------------------
    max_allowed_chat_ids: int = 50  # 每个 project 最多关联的 chat_id 数量
    max_evicted_cache: int = 200  # evicted_chat_ids 有界 LRU 上限
    project_chat_suffix: str = "dev"  # 项目专属群名称后缀
    slock_team_name_suffix: str = Field(
        default="[Slock]",
        description="/new-team 创建 Slock 协作群时追加的名称后缀",
        validation_alias=AliasChoices("slock_team_name_suffix", "slock_team_name_prefix"),
    )

    # Slock 引擎运行参数 --------------------------------------------------------
    slock_reply_mode: str = Field(default="direct", description="Slock 群回复模式: direct（不开话题） / thread（话题回复）")
    slock_max_parallel_agents: int = Field(default=4, ge=1, description="Slock 最大并行 Agent 数（ThreadPool workers）")
    slock_max_queue_size: int = Field(default=8, ge=1, description="Slock 执行队列最大深度，超出时拒绝提交")
    slock_queue_wait_timeout: int = Field(default=60, ge=1, le=600, description="Slock 排队等待超时（秒），超时未执行则取消")
    slock_max_open_tasks: int = Field(default=50, ge=1, description="Slock 单群最大未完成任务数，超出时拒绝创建")
    slock_agent_execution_timeout: int = Field(default=600, ge=30, description="Slock 单 Agent 执行超时（秒），超时后取消 ACP session")
    slock_observer_flush_timeout: int = Field(default=30, ge=5, description="ObserverLearningQueue flush 操作超时（秒）")
    slock_observer_max_queue_size: int = Field(default=10000, ge=100, description="ObserverLearningQueue 最大队列深度，超出时丢弃最旧记录")
    slock_assign_rate_limit: int = Field(default=5, ge=1, description="非管理员每分钟最大任务提交数（rate-limit）")
    slock_escalation_timeout: int = Field(default=1800, ge=60, description="Slock 升级请求自动中止超时（秒），默认 30 分钟")
    slock_idle_scan_interval: int = Field(default=10, ge=1, description="空闲角色扫描 TODO 池间隔（秒）")
    slock_auto_plan_delay: int = Field(default=5, ge=1, le=30, description="任务创建后自动触发规划的延迟（秒）")
    slock_role_evolution_threshold: int = Field(default=3, ge=1, description="角色进化触发阈值（每完成 N 个任务自动更新角色身份描述）")

    # Slock Discussion / NLI / Memory Enhancement ----------------------------
    slock_discussion_enabled: bool = Field(default=True, description="是否启用 Agent 间自动讨论（默认开启）")
    slock_discussion_broadcast_rounds: bool = Field(default=True, description="讨论时逐轮以角色身份卡片单独发送到群")
    slock_discussion_live_card: bool = Field(default=False, description="讨论时是否显示合并式总览卡片（逐轮广播开启时默认关闭）")
    slock_max_discussion_rounds: int = Field(default=3, ge=1, le=10, description="讨论链最大轮次，超限强制收敛")
    slock_uncertainty_markers: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_UNCERTAINTY_MARKERS),
        description="触发讨论的不确定性标记词列表（如 '不确定', 'needs review'）",
    )
    slock_convergence_signals: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CONVERGENCE_SIGNALS),
        description="讨论收敛信号词列表（如 'AGREE', 'LGTM', '同意'）",
    )
    slock_discussion_trigger_rules: str = Field(
        default="coder->reviewer,architect->coder",
        description="讨论触发规则（role->role 格式，逗号分隔）",
    )
    slock_nli_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="NLI 意图分类置信度阈值，低于此值 fallback 到 agent 路由")
    slock_nli_timeout: float = Field(default=2.5, ge=0.1, description="NLI 意图分类超时（秒），超时直接 fallback")
    slock_memory_summarize_threshold: int = Field(default=4000, ge=1000, description="L1 active_context 超过此字符数时触发 LLM 摘要压缩")
    slock_conversation_replay_rounds: int = Field(default=5, ge=1, le=20, description="从 messages.jsonl 回读的最近对话轮数")
    slock_inject_team_roster: bool = Field(default=True, description="构造 agent prompt 时注入同 channel 队友花名册（动态来自 AgentRegistry，不写死角色）")
    slock_team_roster_max_entries: int = Field(default=20, ge=0, le=200, description="花名册最多列出的队友数量，0=禁用")
    slock_default_wake_policy: str = Field(default="smart_judge", description="未设置 agent.wake_policy / channel.wake_policy 时的兜底唤醒策略：on_mention=仅被 @ 才参与路由，smart_judge=按技能与上下文智能判断（兼容现行行为）")
    slock_discussion_token_budget: int = Field(default=50000, ge=1000, description="单次讨论链的 token 预算上限")
    slock_max_parallel_discussions: int = Field(default=3, ge=1, le=10, description="同一 channel 内最大并行讨论数，超出时排队")
    slock_discussion_timeout: int = Field(default=300, ge=30, description="讨论 watchdog 超时秒数，超时自动终止讨论")
    slock_arbiter_max_tokens: int = Field(default=500, ge=100, le=2000, description="讨论最终仲裁者（final arbiter）的最大输出 token 数")
    slock_tool_path_restrictions: list[str] = Field(default_factory=list, description="Slock ACP 工具允许访问的根路径列表")
    slock_dangerous_shell_patterns: list[str] = Field(default_factory=list, description="追加的 Slock shell 危险命令正则")
    slock_memory_summarize_timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="LLM 记忆摘要压缩超时（秒），过短会导致摘要失败退化为截断")
    slock_proactive_followup_enabled: bool = Field(default=True, description="是否允许 Agent 在交付结果后主动跟进")
    slock_proactive_followup_delay: int = Field(default=60, ge=10, le=600, description="结果交付后等待用户反馈的秒数，超时则主动跟进")
    slock_autonomous_task_planning_enabled: bool = Field(
        default=True,
        description="Slock 普通任务消息是否自动进入任务看板和多角色协作规划，而不是直接单 Agent 回复",
    )
    slock_resolution_learning_enabled: bool = Field(default=True, description="是否将自主决策假设持久化以复用（减少重复 LLM 调用）")
    slock_freshness_gate_enabled: bool = Field(default=True, description="发送前检查 channel 是否有新消息（Freshness Gate），防止过期回复")
    slock_freshness_max_retries: int = Field(default=2, ge=0, le=5, description="Freshness Gate 检查不通过时最大重试次数，超出强制发送")
    slock_freshness_reeval_timeout: float = Field(default=15.0, ge=1.0, le=60.0, description="Freshness 重新评估的 LLM 超时（秒）")
    slock_orchestrator_degradation_enabled: bool = Field(default=True, description="协调者不可用时允许执行 agent 自主推进计划")
    slock_semantic_prefilter_enabled: bool = Field(default=True, description="路由前进行 agent 语义自评估，明显不相关的 agent 直接跳过")
    slock_behavior_convergence_enabled: bool = Field(default=True, description="连续失败时自动写入回避策略到 agent 记忆")
    slock_behavior_convergence_threshold: int = Field(default=3, ge=2, le=10, description="连续失败 N 次后触发行为收敛")
    slock_action_card_enabled: bool = Field(default=True, description="副作用操作（shell/commit/delete）强制走 Action Card 确认流")

    # Slock Memory Capacity & Task Chain -------------------------------------------
    slock_l1_max_size: int = Field(default=51200, ge=1024, description="L1 agent 私有记忆最大字节数（默认 50KB），超出触发摘要压缩")
    slock_l2_max_size: int = Field(default=204800, ge=1024, description="L2 group 共享记忆最大字节数（默认 200KB），超出触发 FIFO 截断")
    slock_l3_max_size: int = Field(default=1048576, ge=1024, description="L3 全局 wiki 最大字节数（默认 1MB），超出触发 FIFO 截断")
    slock_chain_templates: str = Field(
        default="coder->reviewer->tester,planner->coder->reviewer->tester",
        description="预置任务协作链模板（role->role->role 格式，逗号分隔多条链）",
        # MIGRATION NOTE (v2.x): Default changed from "coder->reviewer->tester" to include
        # planner chain. For existing deployments, this may trigger auto-planning if planner
        # role exists. The orchestrator guards against creating plans when required roles
        # are missing — no action needed unless you want to opt out (override via env var).
    )
    slock_auto_plan_timeout: int = Field(default=30, ge=5, le=300, description="协作计划自动启动等待秒数（用户无否决则自动执行）")
    slock_role_response_timeout: int = Field(default=120, ge=10, le=300, description="角色主动参与的响应超时秒数，超时则 escalate 或选择次优角色")
    slock_passive_mode: bool = Field(
        default=True,
        description="被动模式：True 时 dispatcher 仅检查 is_managed_chat 即路由消息到 slock，无需 is_slock_active 前置检查",
    )
    slock_default_roles: str = Field(
        default="",
        description=(
            "新建 slock 群时自动创建的预置角色（格式: role:tool_type,role:tool_type）；留空则不自动创建。"
            " 合法 tool_type: codex, claude, coco, aiden, gemini"
        ),
    )

    # Slock Auto-Activation Guard ------------------------------------------------
    slock_auto_activate_whitelist_user_ids: str = Field(
        default="",
        description="被动激活白名单用户 ID（逗号分隔），留空则根据 slock_auto_activate_default_policy 决定权限范围",
    )
    slock_auto_activate_default_policy: str = Field(
        default="allow_all",
        description=(
            "自动激活默认策略：'allow_all'（默认）时白名单为空则允许所有用户触发；"
            "'admin_only' 时白名单为空则仅 admin 可触发"
        ),
    )

    @field_validator("slock_auto_activate_default_policy")
    @classmethod
    def _validate_auto_activate_policy(cls, v: str) -> str:
        valid = {"admin_only", "allow_all"}
        if v not in valid:
            raise ValueError(f"slock_auto_activate_default_policy 必须是 {valid} 之一，当前值: {v!r}")
        return v

    @field_validator("slock_default_wake_policy", mode="before")
    @classmethod
    def _validate_default_wake_policy(cls, v: str) -> str:
        normalized = (v or "smart_judge").strip().lower().replace("-", "_")
        valid = {"on_mention", "smart_judge"}
        if normalized not in valid:
            raise ValueError(f"slock_default_wake_policy 必须是 {valid} 之一，当前值: {v!r}")
        return normalized

    @field_validator("slock_default_roles")
    @classmethod
    def _validate_default_roles(cls, v: str) -> str:
        if not v:
            return v
        valid_tool_types = {"codex", "claude", "coco", "aiden", "gemini"}
        for pair in v.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise ValueError(
                    f"slock_default_roles 格式错误: {pair!r}，应为 role:tool_type"
                )
            _, tool_type = pair.rsplit(":", 1)
            tool_type = tool_type.strip().lower()
            if tool_type not in valid_tool_types:
                raise ValueError(
                    f"slock_default_roles 非法 tool_type: {tool_type!r}，合法值: {valid_tool_types}"
                )
        return v
    slock_auto_activate_rate_limit_per_user: int = Field(
        default=3, ge=1, le=30,
        description="单用户每分钟最大自动激活次数",
    )
    slock_auto_activate_rate_limit_global: int = Field(
        default=10, ge=1, le=100,
        description="全局每分钟最大自动激活次数",
    )
    slock_bootstrap_timeout: int = Field(
        default=10, ge=1, le=60,
        description="默认角色 bootstrap 同步等待超时（秒），超时后队列仍继续消费",
    )
    slock_patrol_interval: int = Field(
        default=30, ge=5, le=300,
        description="Proactive patrol loop 巡检间隔（秒），检查 SLA 超时和空闲 Agent 认领",
    )
    slock_task_default_sla: int = Field(
        default=300, ge=30, le=3600,
        description="任务默认 SLA 超时（秒），超时后巡检器发送提醒并尝试重新分配",
    )
    slock_session_affinity_window: int = Field(
        default=120, ge=10, le=600,
        description="会话粘性窗口（秒），同一用户连续消息优先路由给同一 Agent",
    )

    @field_validator("slock_team_name_suffix", mode="before")
    @classmethod
    def _warn_deprecated_slock_prefix(cls, v: str) -> str:
        """Emit deprecation warning when the old alias is used."""
        # The warning fires unconditionally here; the alias mapping is handled
        # by Pydantic's validation_alias. We detect usage of the old env var
        # by checking os.environ for the deprecated key name.
        if os.environ.get("SLOCK_TEAM_NAME_PREFIX") is not None:
            _emit_slock_prefix_deprecation_warning()
        return v

    @field_validator("max_allowed_chat_ids", mode="before")
    @classmethod
    def _max_allowed_chat_ids_must_be_positive(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name.upper()} 必须 ≥ 1（当前值: {v}）")
        return int(v)

    @field_validator("lock_confirm_timeout", "max_evicted_cache", mode="before")
    @classmethod
    def _lock_confirm_and_evicted_cache_must_be_positive(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name.upper()} 必须 > 0（当前值: {v}）")
        return int(v)

    @field_validator("repo_lock_idle_timeout", "repo_lock_cleanup_interval", "repo_lock_hard_timeout", mode="before")
    @classmethod
    def _repo_lock_timers_must_be_positive(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name.upper()} 必须 > 0（当前值: {v}）")
        return int(v)

    @field_validator("chat_lock_max_duration", "chat_lock_cleanup_interval", mode="before")
    @classmethod
    def _chat_lock_timers_must_be_positive(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name.upper()} 必须 > 0（当前值: {v}）")
        return int(v)

    @field_validator("lock_undo_window_seconds", mode="before")
    @classmethod
    def _lock_undo_window_seconds_in_range(cls, v: int, info) -> int:
        val = int(v)
        if val < 60 or val > 600:
            raise ValueError(
                f"LOCK_UNDO_WINDOW_SECONDS 必须在 [60, 600] 范围内（秒）（当前值: {v}）"
            )
        if val % 60 != 0:
            raise ValueError(
                f"LOCK_UNDO_WINDOW_SECONDS 必须为 60 的整数倍（当前值: {val}），"
                "可选值如 60, 120, 180, 240, 300, …"
            )
        return val

    @field_validator(
        "spec_review_timeout", "spec_review_min_timeout", "spec_review_hard_floor",
        "spec_review_startup_timeout",
        mode="before",
    )
    @classmethod
    def _spec_review_timeout_fields_must_be_positive(cls, v: int, info) -> int:
        val = int(v)
        if val < 1:
            raise ValueError(f"{info.field_name} 必须 > 0，当前值为 {v}")
        return val

    @field_validator("spec_review_max_parallel", mode="before")
    @classmethod
    def _spec_review_max_parallel_must_be_in_range(cls, v: int, info) -> int:
        val = int(v)
        if val < 1:
            raise ValueError(f"{info.field_name} 必须 ≥ 1，当前值为 {v}")
        if val > 20:
            raise ValueError(f"{info.field_name} 必须 ≤ 20，当前值为 {v}")
        return val

    @field_validator(
        "spec_review_dynamic_roles_max",
        "spec_review_total_roles_max",
        "spec_review_pass_streak_required",
        mode="before",
    )
    @classmethod
    def _spec_review_adaptive_fields_must_be_positive(cls, v: int, info) -> int:
        val = int(v)
        if val < 1:
            raise ValueError(f"{info.field_name} 必须 > 0，当前值为 {v}")
        if info.field_name == "spec_review_total_roles_max" and val < 5:
            raise ValueError(f"{info.field_name} 必须 ≥ 5，当前值为 {v}")
        return val

    @field_validator("spec_review_retry_max_delay", mode="before")
    @classmethod
    def _spec_retry_max_delay_must_be_positive(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name} 必须 > 0，当前值为 {v}")
        return int(v)

    @field_validator("spec_review_retry_max_attempts", mode="before")
    @classmethod
    def _spec_retry_max_attempts_must_be_non_negative(cls, v: int, info) -> int:
        val = int(v)
        if val < 0:
            raise ValueError(
                f"{info.field_name} 必须 ≥ 0（设为 0 可禁用重试），当前值为 {v}"
            )
        # 上限 10：单次 retry 耗时 ≈ max_delay + adaptive_timeout * multiplier，
        # 10 次重试可能导致单 cycle 总耗时超过 cycle budget，不建议生产使用（推荐 1-3）。
        if val > 10:
            raise ValueError(
                f"{info.field_name} 必须 ≤ 10（推荐 1-3），当前值为 {v}"
            )
        return val

    @field_validator("spec_review_role_timeout_multipliers", mode="before")
    @classmethod
    def _clamp_role_timeout_multipliers(cls, v, info) -> dict[str, float]:
        import json as _json

        if isinstance(v, str):
            v = _json.loads(v)
        if not isinstance(v, dict):
            raise ValueError(f"{info.field_name} 必须是 dict[str, float]")
        clamped: dict[str, float] = {}
        for k, val in v.items():
            fval = float(val)
            if fval < 0.1:
                fval = 0.1
            if fval > 3.0:
                fval = 3.0
            clamped[str(k)] = fval
        return clamped

    @model_validator(mode="before")
    @classmethod
    def _hoist_card_fields(cls, data: dict) -> dict:
        """Collect flat card_* env keys into nested 'card' sub-dict for CardSessionConfig."""
        if not isinstance(data, dict):
            return data
        # If 'card' is already a dict/model, skip hoisting (e.g. programmatic construction)
        if "card" in data and isinstance(data["card"], (dict, CardSessionConfig)):
            return data
        # Map from flat Settings field name (card_xxx) to CardSessionConfig field name (xxx)
        _CARD_FIELD_MAP = {
            "card_collapsible_enabled": "collapsible_enabled",
            "card_continuation_enabled": "continuation_enabled",
            "card_button_layout": "button_layout",
            "card_button_size": "button_size",
            "card_mobile_force_vertical": "mobile_force_vertical",
            "card_deep_compact_default": "deep_compact_default",
            "card_max_chars": "max_chars",
            "card_session_lock_max": "session_lock_max",
            "card_session_lock_ttl": "session_lock_ttl",
            "card_session_idle_timeout": "session_idle_timeout",
            "card_session_idle_warn_before": "session_idle_warn_at_remaining",
            "card_session_idle_warn_at_remaining": "session_idle_warn_at_remaining",
            "card_session_max_rotations": "session_max_rotations",
            "card_delivery_pool_max_workers": "delivery_pool_max_workers",
            "card_delivery_api_timeout": "delivery_api_timeout",
            "card_action_dedup_ttl": "action_dedup_ttl",
            "card_action_dedup_max_size": "action_dedup_max_size",
            "card_action_dedup_cleanup_interval": "action_dedup_cleanup_interval",
            "card_ticker_interval": "ticker_interval",
            "card_task_level_cards_enabled": "task_level_cards_enabled",
            "card_max_task_cards": "max_task_cards",
        }
        card_data: dict = {}
        for flat_key, nested_key in _CARD_FIELD_MAP.items():
            if flat_key in data:
                card_data[nested_key] = data.pop(flat_key)
        if card_data:
            data["card"] = card_data
        return data

    @model_validator(mode="after")
    def _validate_spec_review_cross_fields(self) -> "Settings":
        """Cross-field validation for spec review timing parameters."""
        # 排序约束: hard_floor <= min_timeout <= timeout
        if self.spec_review_hard_floor > self.spec_review_min_timeout:
            raise ValueError(
                f"spec_review_hard_floor 必须 ≤ spec_review_min_timeout，"
                f"当前分别为 {self.spec_review_hard_floor} 和 {self.spec_review_min_timeout}"
            )
        if self.spec_review_min_timeout > self.spec_review_timeout:
            raise ValueError(
                f"spec_review_min_timeout 必须 ≤ spec_review_timeout，"
                f"当前分别为 {self.spec_review_min_timeout} 和 {self.spec_review_timeout}"
            )
        # 重试最大延迟不能超过审查超时
        if self.spec_review_retry_max_delay > self.spec_review_timeout:
            raise ValueError(
                f"spec_review_retry_max_delay 必须 ≤ spec_review_timeout，"
                f"当前分别为 {self.spec_review_retry_max_delay} 和 {self.spec_review_timeout}"
            )
        # 下界估算：实际每次 retry 耗时由 compute_adaptive_timeout 动态决定，
        # 可能大于 min_timeout；此处使用 min_timeout 作为保守下界验证总预算合理性。
        total_retry_budget = (
            self.spec_review_retry_max_delay + self.spec_review_min_timeout
        ) * self.spec_review_retry_max_attempts
        budget_limit = self.spec_review_timeout * 2
        if total_retry_budget > budget_limit:
            raise ValueError(
                "请减小 SPEC_REVIEW_RETRY_MAX_ATTEMPTS 或 SPEC_REVIEW_RETRY_MAX_DELAY"
                "（当前组合超出允许范围）"
            )
        # NOTE: realistic budget check moved to _post_validate_warnings()
        return self

    @model_validator(mode="after")
    def _validate_lock_timing_cross_fields(self) -> "Settings":
        """Cross-field: lock_undo_window_seconds should be >= lock_confirm_timeout."""
        if self.lock_undo_window_seconds < self.lock_confirm_timeout:
            _logging.getLogger(__name__).warning(
                "lock_undo_window_seconds (%d) < lock_confirm_timeout (%d): "
                "confirmation timeout exceeds undo window, which may confuse users. "
                "Consider increasing lock_undo_window_seconds or decreasing lock_confirm_timeout.",
                self.lock_undo_window_seconds, self.lock_confirm_timeout,
            )
        return self

    @model_validator(mode="after")
    def _log_slock_default_roles_empty(self) -> "Settings":
        """Note when optional automatic role provisioning is disabled."""
        if not self.slock_default_roles:
            _logging.getLogger(__name__).info(
                "slock_default_roles is empty. No preset roles will be auto-created when creating new slock groups. "
                "Configure SLOCK_DEFAULT_ROLES in .env if you want automatic role provisioning "
                "(e.g., 'planner:claude,coder:codex,reviewer:claude,tester:codex')."
            )
        return self

    @property
    def command_blacklist(self) -> list[str]:
        return [cmd.strip() for cmd in self.sandbox_command_blacklist.split(",") if cmd.strip()]

    @property
    def command_whitelist(self) -> list[str]:
        return [cmd.strip() for cmd in self.sandbox_command_whitelist.split(",") if cmd.strip()]

    def validate_feishu_config(self) -> bool:
        return bool(self.app_id and self.app_secret)

    def get_acp_command(self, agent_type: str) -> tuple[str, list[str]]:
        """Return (cmd, args) override for an ACP agent, if configured."""
        agent_type = (agent_type or "").lower()
        if agent_type == "coco" and self.coco_acp_cmd:
            return self.coco_acp_cmd, shlex.split(self.coco_acp_args or "")
        if agent_type == "claude" and self.claude_acp_cmd:
            return self.claude_acp_cmd, shlex.split(self.claude_acp_args or "")
        return "", []
