"""Configuration package — split from monolithic config.py for maintainability.

All public names are re-exported here so existing ``from src.config import …``
statements continue to work without modification.
"""

from .card import CardSessionConfig
from .errors import ConfigurationError
from .security_posture import (
    IngressAccessMode,
    SecurityFinding,
    SecurityPosture,
    SecuritySeverity,
    ShellAccessMode,
    evaluate_security_posture,
)
from .settings import Settings
from .singleton import (
    _post_validate_warnings,
    get_settings,
)
from .spec import SpecReviewConfig

__all__ = [
    "CardSessionConfig",
    "ConfigurationError",
    "IngressAccessMode",
    "SecurityFinding",
    "SecurityPosture",
    "SecuritySeverity",
    "Settings",
    "ShellAccessMode",
    "SpecReviewConfig",
    "evaluate_security_posture",
    "get_settings",
    "_post_validate_warnings",
]
