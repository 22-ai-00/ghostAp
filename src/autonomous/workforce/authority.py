"""Journal-projected workforce authority state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuthorityMode(str, Enum):
    LEGACY_WRITE = "legacy_write"
    SHADOW_READ = "shadow_read"
    V5_WRITE = "v5_write"
    V5_ONLY = "v5_only"


@dataclass(frozen=True)
class AuthoritySnapshot:
    epoch: int
    mode: AuthorityMode
    cutover_sequence: int = 0


class StaleAuthorityEpoch(RuntimeError):
    """A stale authority epoch cannot mutate canonical workforce state."""


__all__ = ["AuthorityMode", "AuthoritySnapshot", "StaleAuthorityEpoch"]
