"""Singleton management for the Settings instance."""

import logging as _logging
import threading
from typing import Optional

from pydantic import ValidationError

from .card import CardSessionConfig
from .errors import ConfigurationError
from .settings import Settings

_settings: Optional[Settings] = None
_settings_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _post_validate_warnings(settings: Settings) -> None:
    """Emit soft-constraint warnings AFTER Settings construction (not inside validator)."""
    logger = _logging.getLogger(__name__)
    try:
        budget_limit = settings.spec_review_timeout * 2
        realistic_budget = (
            settings.spec_review_retry_max_delay + settings.spec_review_timeout
        ) * settings.spec_review_retry_max_attempts
        if realistic_budget > budget_limit and settings.spec_review_retry_max_attempts > 0:
            logger.info(
                "重试实际预算可能超限：(retry_max_delay + base_timeout) × max_attempts = %d，"
                "超过 timeout × 2 = %d。首次 retry 耗时可能远超预期",
                realistic_budget, budget_limit,
            )
    except (TypeError, AttributeError):
        pass  # gracefully handle mock/proxy settings objects

    # Card Session config summary
    try:
        logger.info(
            "CardSession config: idle_timeout=%ds, lock_ttl=%ds, max_rotations=%d, continuation=%s",
            settings.card.session_idle_timeout,
            settings.card.session_lock_ttl,
            settings.card.session_max_rotations,
            settings.card.continuation_enabled,
        )
    except (TypeError, AttributeError):
        pass


def _build_spec_review_recommended_hint() -> str:
    """Dynamically build recommended combination from Settings field defaults."""
    fields_of_interest = [
        "spec_review_hard_floor",
        "spec_review_min_timeout",
        "spec_review_timeout",
        "spec_review_retry_max_delay",
        "spec_review_retry_max_attempts",
    ]
    parts = []
    for name in fields_of_interest:
        fi = Settings.model_fields.get(name)
        if fi and fi.default is not None:
            parts.append(f"{name.upper()}={fi.default}")
    if not parts:
        return ""
    return f"\n（推荐组合: {', '.join(parts)}）"


def _build_card_session_recommended_hint() -> str:
    """Build recommended combination for Card Session cross-field errors."""
    fields_of_interest = [
        "session_lock_ttl",
        "session_idle_timeout",
    ]
    parts = []
    for name in fields_of_interest:
        fi = CardSessionConfig.model_fields.get(name)
        if fi and fi.default is not None:
            parts.append(f"CARD_{name.upper()}={fi.default}")
    if not parts:
        return ""
    return f"\n（推荐组合: {', '.join(parts)}）"


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                try:
                    _settings = Settings()
                except Exception as e:
                    # Friendly error for pydantic ValidationError
                    if isinstance(e, ValidationError):
                        errors = e.errors()
                        lines = ["配置错误:"]
                        for err in errors:
                            loc = ".".join(str(x) for x in err.get("loc", []))
                            msg = err.get("msg", "")
                            # Attempt to show field default value for recovery guidance
                            default_hint = ""
                            if loc and loc in Settings.model_fields:
                                fi = Settings.model_fields[loc]
                                _default = fi.default
                                if _default is not None and str(_default) != "PydanticUndefined":
                                    default_hint = f"（默认值: {_default}）"
                            # For cross-field validation errors, provide recommended combination
                            if not default_hint and ("value_error" in err.get("type", "") or not loc):
                                # Detect card_session vs spec_review errors
                                loc_lower = loc.lower()
                                if "card_session" in loc_lower or "card_session" in msg.lower():
                                    default_hint = _build_card_session_recommended_hint()
                                else:
                                    default_hint = _build_spec_review_recommended_hint()
                            lines.append(f"  - {(loc.upper() or '[跨字段校验]')}: {msg} {default_hint}".rstrip())
                        lines.append("建议: 检查 .env 文件中对应配置项的值是否合法")
                        raise ConfigurationError("\n".join(lines)) from e
                    raise
                _post_validate_warnings(_settings)
    return _settings
