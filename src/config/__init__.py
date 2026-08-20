"""Configuration package — split from monolithic config.py for maintainability.

All public names are re-exported here so existing ``from src.config import …``
statements continue to work without modification.
"""

from .access import (
    AccessFinding,
    AccessFindingSeverity,
    IngressAccessMode,
)
from .card import CardSessionConfig
from .errors import ConfigurationError
from .settings import Settings
from .singleton import (
    _post_validate_warnings,
    get_settings,
)

__all__ = [
    "CardSessionConfig",
    "ConfigurationError",
    "IngressAccessMode",
    "AccessFinding",
    "AccessFindingSeverity",
    "Settings",
    "get_settings",
    "_post_validate_warnings",
]
