"""Canonical identifiers and immutable-value helpers for autonomous domains."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def new_id(prefix: str) -> str:
    """Return a stable-prefix random identifier."""
    if not prefix:
        raise ValueError("identifier prefix is required")
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def freeze(value: Any) -> Any:
    """Recursively convert JSON-like values to immutable equivalents."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Convert immutable domain values back to JSON-compatible structures."""
    if isinstance(value, Mapping):
        return {str(key): thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(thaw(item) for item in value)
    return value


def strict_bool(value: Any, field_name: str) -> bool:
    """Accept only literal booleans at untrusted serialization boundaries."""
    if value is not True and value is not False:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def strict_int(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
) -> int:
    """Accept only literal integers, never booleans or numeric strings."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def strict_float(value: Any, field_name: str) -> float:
    """Accept finite numeric values without bool or string coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def strict_str(value: Any, field_name: str) -> str:
    """Accept only literal strings at untrusted serialization boundaries."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value
