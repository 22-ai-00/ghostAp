"""CardSessionConfig — nested configuration for card session / delivery / UI parameters."""

import logging as _logging
import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CardSessionConfig(BaseModel):
    """Nested configuration for card session / delivery / UI parameters.

    All fields map 1:1 to CARD_* environment variables. The parent Settings
    class uses a model_validator to hoist flat card_* fields into this nested
    model while keeping .env backward compatibility.
    """

    continuation_enabled: bool = True
    button_size: Literal["small", "medium", "large"] = "medium"
    mobile_force_vertical: bool = True
    max_chars: int = 28000
    session_lock_max: int = 10_000
    session_lock_ttl: float = 600.0
    session_idle_timeout: int = 1800
    session_idle_warn_at_remaining: int = 300
    session_max_rotations: int = 20
    action_dedup_ttl: int = 1
    action_dedup_max_size: int = 5000
    delivery_pool_max_workers: int = 16
    delivery_api_timeout: float = Field(
        default=35.0,
        description="Feishu card API hard timeout in seconds; should exceed the lark SDK timeout slightly.",
    )
    ticker_interval: float = Field(
        default=1.2,
        gt=0,
        description="Live ticker 帧切换间隔（秒），对应 v2 设计中绿点动画节奏",
    )
    build_heartbeat_interval: float = Field(
        default=5.0,
        gt=0,
        description="Spec BUILD 阶段心跳间隔（秒），定期刷新 footer 进度文本",
    )
    task_level_cards_enabled: bool = Field(
        default=False,
        description="启用后多步任务使用独立飞书卡片展示每个子任务；默认关闭以避免飞书消息刷屏",
    )
    max_task_cards: int = Field(
        default=8,
        description="单次执行中任务级卡片数量上限，超出后合并到最后一张卡片",
    )

    @field_validator(
        "max_task_cards",
        "session_lock_max",
        "max_chars",
        "session_idle_timeout",
        "session_idle_warn_at_remaining",
        "session_max_rotations",
        "delivery_pool_max_workers",
        "action_dedup_ttl",
        "action_dedup_max_size",
        mode="before",
    )
    @classmethod
    def _bounded_int(cls, v: int, info) -> int:
        bounds = {
            "max_task_cards": (1, 20),
            "session_lock_max": (1000, 100_000),
            "max_chars": (1000, 50_000),
            "session_idle_timeout": (300, 7200),
            "session_idle_warn_at_remaining": (60, 3600),
            "session_max_rotations": (1, 100),
            "delivery_pool_max_workers": (1, 32),
            "action_dedup_ttl": (0, 10),
            "action_dedup_max_size": (100, 50_000),
        }
        value = int(v)
        lower, upper = bounds[info.field_name]
        if value < lower or value > upper:
            raise ValueError(
                f"card_{info.field_name} 必须在 [{lower}, {upper}] 范围内（当前值: {v}）"
            )
        return value

    @field_validator("session_lock_ttl", "delivery_api_timeout", mode="before")
    @classmethod
    def _bounded_float(cls, v: float, info) -> float:
        val = float(v)
        lower, upper = (
            (60, 3600) if info.field_name == "session_lock_ttl" else (1, 300)
        )
        if val < lower or val > upper:
            raise ValueError(
                f"card_{info.field_name} 必须在 [{lower}, {upper}] 范围内（当前值: {v}）"
            )
        if info.field_name == "session_lock_ttl" and val % 60 != 0:
            new_val = math.ceil(val / 60) * 60
            _logging.getLogger(__name__).info(
                "CARD_SESSION_LOCK_TTL rounded up to %ds (from %s)", new_val, v
            )
            val = float(new_val)
        return val

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "CardSessionConfig":
        """Cross-field: session_lock_ttl must not exceed session_idle_timeout.
        session_idle_warn_at_remaining must be less than session_idle_timeout."""
        if self.session_lock_ttl > self.session_idle_timeout:
            raise ValueError(
                f"card_session_lock_ttl 必须 ≤ card_session_idle_timeout（秒），"
                f"当前分别为 {self.session_lock_ttl} 和 {self.session_idle_timeout}"
            )
        if self.session_idle_warn_at_remaining >= self.session_idle_timeout:
            raise ValueError(
                f"card_session_idle_warn_at_remaining 必须 < card_session_idle_timeout（秒），"
                f"当前分别为 {self.session_idle_warn_at_remaining} 和 {self.session_idle_timeout}"
            )
        return self
