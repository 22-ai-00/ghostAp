"""Settings — main application configuration model backed by pydantic-settings."""

import base64
import binascii
import logging as _logging
import os
import shlex
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

from .card import CardSessionConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_id: str = ""
    app_secret: str = ""

    # Default ACP tool for SMART mode (e.g., "coco", "claude", "aiden", "codex", "gemini", "traex", "grok", "dsh")
    # When set, unmatched messages in SMART mode are forwarded to this tool.
    # When empty, all unmatched messages are treated as shell commands.
    default_acp_tool: str = ""

    # Reliability limits for direct host command execution.
    command_timeout: int = 30
    command_max_output_length: int = 4000
    ingress_access_mode: Literal[
        "enforced", "shadow", "legacy_allow_all"
    ] = "enforced"
    admin_bootstrap_scope: Literal["any_chat", "p2p_only"] = "p2p_only"
    employee_group_context_retention_days: int = Field(
        default=30,
        ge=1,
        le=3650,
    )

    coco_execution_timeout: int = Field(default=7200, gt=0)
    coco_session_timeout: int = 86400

    claude_execution_timeout: int = Field(default=7200, gt=0)
    claude_session_timeout: int = 86400
    programming_finalization_reserve_s: int = Field(
        default=600,
        ge=0,
        le=3600,
        description=(
            "Seconds reserved from ordinary programming execution time for a "
            "separate bounded finalization prompt; 0 disables the reserve."
        ),
    )
    programming_max_execution_windows: int = Field(
        default=4,
        ge=1,
        le=24,
        description=(
            "Maximum bounded ACP execution windows for one ordinary "
            "programming task. Timeout-finalized incomplete work resumes on "
            "a fresh transport between windows."
        ),
    )
    programming_agent_idle_timeout_s: int = Field(
        default=420,
        ge=0,
        description=(
            "Activity-based idle timeout for ordinary programming ACP turns. "
            "ACP text, thought, plan, and tool events refresh the timer while "
            "the execution timeout remains the hard cap. Set 0 to disable."
        ),
    )
    feishu_cot_enabled: bool = Field(
        default=True,
        description=(
            "Use Feishu IM COT/AG-UI for ordinary programming process events. "
            "Task summaries and terminal conclusions remain Card 2.0 messages; "
            "unsupported tenants fall back to the existing streaming card."
        ),
    )
    feishu_cot_detail: Literal["brief", "detailed"] = Field(
        default="brief",
        description=(
            "Tool detail carried by Feishu COT process messages. Brief keeps "
            "tool actions and terminal status without raw arguments/output; "
            "detailed includes sanitized bounded arguments and output."
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
    # degrading. With lazy startup, the first explicit model-catalog request may
    # pay this bounded cost; successful results keep later interactions cached.
    acp_model_probe_timeout: float = 15.0

    # ACP backends are lazy by default: Bot readiness must not depend on model
    # provider or package-registry network availability. Operators can opt in
    # to eager model-catalog warming when desired.
    acp_model_preheat_on_startup: bool = False

    # Auto-update agent CLI when ACP server mode is not supported
    acp_auto_update: bool = True
    # Timeout for agent CLI auto-update subprocess (seconds)
    acp_auto_update_timeout: int = 120

    # Engine eval prompt timeout (seconds) — used by Spec engine
    engine_eval_prompt_timeout: int = 60

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

    # Subagent encouragement hint: when True (default), every
    # agent() prompt template and every script-generation prompt template
    # includes a trailing paragraph that encourages the model to delegate to
    # subagents. Set to False to suppress the hint
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
        default=600,
        ge=10,
        description="AI workflow script generation attempt timeout (seconds), bounded by the shared 600-second generation deadline.",
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

    # 审查会话启动超时（秒），独立于全局 acp_startup_timeout
    spec_review_startup_timeout: int = 30

    # ---- 完成度把控（Completion Control）----
    spec_objective_verify_enabled: bool = True
    spec_objective_verify_timeout: int = 300
    spec_completion_gate_enabled: bool = True

    # Card session / delivery / UI configuration (nested model)
    card: CardSessionConfig = CardSessionConfig()

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
    task_scheduler_max_pending_normal: int = Field(default=1000, gt=0)
    task_scheduler_max_pending_system: int = Field(default=100, gt=0)
    task_scheduler_max_terminal_history: int = Field(default=5000, gt=0)
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
        allow_inf_nan=False,
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

    # 消息回复模式配置
    # - direct: 直接回复（消息显示在被回复消息下方）
    # - thread: 话题回复（使用 reply_in_thread=True，消息会显示在独立话题区域，更整洁）
    #
    # default_reply_mode: 其他模式（Coco/Claude/Shell/Deep等）的回复方式（默认 thread，话题回复更整洁）
    default_reply_mode: str = "thread"

    thread_programming_enabled: bool = True

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

    # ------------------------------------------------------------------
    # 管理员用户列表（用于群级锁权限判定）
    # Stored as frozenset for O(1) membership checks on hot paths.
    # Declared as str to prevent pydantic-settings from attempting JSON parse
    # on plain comma-separated values; converted to frozenset in model_validator.
    # ------------------------------------------------------------------
    admin_user_ids: str = ""

    # ------------------------------------------------------------------
    # Employee Department persistence
    # ------------------------------------------------------------------
    autonomous_state_dir: str = "~/.ghostap/autonomy"
    autonomous_journal_dir: str = "~/.ghostap/autonomy/journal"
    autonomous_journal_hmac_key: SecretStr = SecretStr("")
    # Team employee data-continuity ABI. Keep this root stable so existing
    # employee workspaces remain discoverable across upgrades.
    autonomous_employee_storage_base: str = "~/.ghostap/slock"
    # Team employee credential continuity ABI. Credentials remain below the
    # established employee storage root and must not be relocated implicitly.
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
    autonomous_employee_ingress_blob_dir: str = (
        "~/.ghostap/autonomy/ingress-blobs"
    )
    autonomous_employee_outbox_blob_dir: str = (
        "~/.ghostap/autonomy/outbox-blobs"
    )
    autonomous_main_bot_warning_blob_dir: str = (
        "~/.ghostap/autonomy/main-bot-warning-blobs"
    )
    autonomous_employee_attachment_staging_dir: str = (
        "~/.ghostap/autonomy/employee-attachments"
    )
    autonomous_employee_system_prompt_token_reserve: int = Field(
        default=4096,
        ge=1,
        le=1_000_000,
    )
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
    autonomous_anchor_path: str = "~/.ghostap/autonomy/journal.anchor"
    autonomous_visible_employee_limit: int = Field(default=8, ge=0)
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

    @field_validator(
        "autonomous_main_bot_audit_dir",
        "autonomous_main_bot_audit_anchor_path",
    )
    @classmethod
    def _validate_main_bot_audit_paths(cls, value: str) -> str:
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("main Bot audit paths must be text without NUL")
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

    @field_validator(
        "max_allowed_chat_ids",
        "lock_confirm_timeout",
        "max_evicted_cache",
        "repo_lock_idle_timeout",
        "repo_lock_cleanup_interval",
        "repo_lock_hard_timeout",
        "chat_lock_max_duration",
        "chat_lock_cleanup_interval",
        mode="before",
    )
    @classmethod
    def _positive_int(cls, v: int, info: ValidationInfo) -> int:
        value = int(v)
        if value < 1:
            raise ValueError(f"{info.field_name.upper()} 必须 > 0（当前值: {v}）")
        return value

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
        "spec_review_timeout",
        "spec_review_min_timeout",
        "spec_review_hard_floor",
        "spec_review_startup_timeout",
        "spec_review_max_parallel",
        "spec_review_dynamic_roles_max",
        "spec_review_total_roles_max",
        "spec_review_pass_streak_required",
        "spec_review_retry_max_delay",
        "spec_review_retry_max_attempts",
        mode="before",
    )
    @classmethod
    def _bounded_spec_review_int(cls, v: int, info: ValidationInfo) -> int:
        bounds = {
            "spec_review_max_parallel": (1, 20),
            "spec_review_total_roles_max": (5, None),
            "spec_review_retry_max_attempts": (0, 10),
        }
        lower, upper = bounds.get(info.field_name, (1, None))
        value = int(v)
        if value < lower or (upper is not None and value > upper):
            ceiling = f", {upper}" if upper is not None else ""
            raise ValueError(
                f"{info.field_name} 必须在 [{lower}{ceiling}] 范围内，当前值为 {v}"
            )
        return value

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
            "card_continuation_enabled": "continuation_enabled",
            "card_button_size": "button_size",
            "card_mobile_force_vertical": "mobile_force_vertical",
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
