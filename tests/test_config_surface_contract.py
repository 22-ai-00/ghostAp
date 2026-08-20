"""Contracts keeping GhostAP's operator-facing configuration surface honest."""

from __future__ import annotations

import re
from pathlib import Path

import src.config as config_package
from src.config import CardSessionConfig, Settings

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_ENV_ASSIGNMENT = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*)=")

# These names are deployment aliases for Settings fields rather than separate
# product settings. Keep the exception list explicit so stale feature flags do
# not become valid merely by acquiring a GHOSTAP_ prefix.
_DEPLOYMENT_ENV_ALIASES = frozenset(
    {
        "GHOSTAP_RESTART_GATE_DIR",
        "GHOSTAP_RESTART_GATE_TIMEOUT",
    }
)

# Only keys owned by GhostAP are governed by this contract. Generic process or
# third-party variables may be documented alongside them without being mistaken
# for Settings fields.
_GHOSTAP_ENV_PREFIXES = (
    "ACP_",
    "ADMIN_",
    "APP_",
    "AUTONOMOUS_",
    "CARD_",
    "CHAT_",
    "CLAUDE_",
    "COCO_",
    "COMMAND_",
    "DEEP_",
    "DEFAULT_",
    "EMPLOYEE_",
    "ENGINE_",
    "FEISHU_",
    "GHOSTAP_",
    "IM_",
    "INGRESS_",
    "LOCK_",
    "MAX_",
    "MESSAGE_",
    "MODEL_",
    "PROGRAMMING_",
    "PROJECT_",
    "RATE_",
    "REPO_",
    "SANDBOX_",
    "SHELL_",
    "SIG_",
    "SMART_",
    "SPEC_",
    "STREAMING_",
    "SYSTEM_",
    "TASK_",
    "THREAD_",
    "WORKFLOW_",
)

_RETIRED_SETTINGS_FIELDS = frozenset(
    {
        "employee_department_enabled",
        "engine_aux_prompt_timeout",
        "acp_permission_auto_approve",
        "acp_trusted_personal_ack",
        "acp_trusted_personal_mode",
        "claude_cli_skip_permissions",
        "project_allowed_roots",
        "review_circuit_lint_fallback_enabled",
        "review_circuit_lint_timeout",
        "review_circuit_success_rate_threshold",
        "review_circuit_window_size",
        "review_metrics_exporter_type",
        "review_metrics_jsonl_path",
        "sandbox_command_blacklist",
        "sandbox_command_whitelist",
        "sandbox_max_output_length",
        "sandbox_strict_lock_mode",
        "sandbox_timeout",
        "sandbox_use_whitelist",
        "shell_access_mode",
        "shell_trusted_local_ack",
        "spec_review_parse_failure_default",
        "spec_review_retry_decay_factor",
        "spec_review_strategy",
    }
)


def _active_example_keys() -> frozenset[str]:
    keys: set[str] = set()
    for line in (_REPOSITORY_ROOT / ".env.example").read_text(
        encoding="utf-8"
    ).splitlines():
        match = _ACTIVE_ENV_ASSIGNMENT.match(line.strip())
        if match is not None:
            keys.add(match.group("key"))
    return frozenset(keys)


def _schema_env_keys() -> frozenset[str]:
    settings_keys = {name.upper() for name in Settings.model_fields}
    card_keys = {
        f"CARD_{name.upper()}" for name in CardSessionConfig.model_fields
    }
    return frozenset(settings_keys | card_keys | set(_DEPLOYMENT_ENV_ALIASES))


def test_env_example_has_no_ghostap_keys_ignored_by_the_schema() -> None:
    active_ghostap_keys = {
        key
        for key in _active_example_keys()
        if key.startswith(_GHOSTAP_ENV_PREFIXES)
    }
    ignored = active_ghostap_keys - _schema_env_keys()

    assert ignored == set(), (
        ".env.example contains active GhostAP keys that Settings silently ignores: "
        f"{sorted(ignored)}"
    )


def test_env_example_does_not_expose_dead_employee_department_switch() -> None:
    assert "EMPLOYEE_DEPARTMENT_ENABLED" not in _active_example_keys()


def test_retired_internal_tuning_fields_are_not_public_settings() -> None:
    leaked = _RETIRED_SETTINGS_FIELDS & Settings.model_fields.keys()

    assert leaked == set(), (
        "retired implementation knobs remain in Settings.model_fields: "
        f"{sorted(leaked)}"
    )


def test_legacy_spec_review_view_is_not_public_configuration() -> None:
    leaked_exports = {
        name
        for name in ("Settings.spec_review", "src.config.SpecReviewConfig")
        if (
            (name == "Settings.spec_review" and hasattr(Settings, "spec_review"))
            or (
                name == "src.config.SpecReviewConfig"
                and (
                    hasattr(config_package, "SpecReviewConfig")
                    or "SpecReviewConfig" in config_package.__all__
                )
            )
        )
    }

    assert leaked_exports == set(), (
        "legacy Spec review configuration remains public: "
        f"{sorted(leaked_exports)}"
    )
