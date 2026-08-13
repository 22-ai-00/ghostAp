"""Environment variable utilities for subprocess spawning."""

from __future__ import annotations

import getpass
import os
import sys
import tempfile
import threading
from typing import Callable, Optional

# Keys that must be removed to avoid nested-session guard crashes
# when spawning ACP / CLI subprocesses from within a wrapper environment.
_GUARD_KEYS = ("CLAUDECODE",)

# Type alias for test environment checker function
TestEnvironmentChecker = Callable[[], bool]

# Global state for injected test environment checker
_test_environment_checker: Optional[TestEnvironmentChecker] = None
_test_environment_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _default_is_test_environment() -> bool:
    """Default implementation for checking if we're running in a test environment.

    Returns:
        True if running in pytest or a test environment, False otherwise.
    """
    # Check for pytest-specific environment variables
    if os.getenv("PYTEST_CURRENT_TEST") is not None:
        return True

    # Check if pytest module is loaded
    if "pytest" in sys.modules:
        return True

    # Check for common test environment markers
    test_markers = ("TESTING", "TEST", "CI")
    for marker in test_markers:
        value = os.getenv(marker)
        if value is not None:
            lower_value = value.strip().lower()
            if lower_value not in ("", "0", "false", "no", "off"):
                return True

    return False


def is_test_environment() -> bool:
    """Check if we're running in a test environment.

    This function supports dependency injection: use `set_test_environment_checker()`
    to replace the default implementation.

    Returns:
        True if running in pytest or a test environment, False otherwise.
    """
    with _test_environment_lock:
        if _test_environment_checker is not None:
            return _test_environment_checker()
    return _default_is_test_environment()


def set_test_environment_checker(checker: Optional[TestEnvironmentChecker]) -> None:
    """Set a custom test environment checker function.

    This allows for dependency injection to support different test environments
    (unit tests, integration tests, E2E tests).

    Args:
        checker: Custom function that returns True if in a test environment.
                 Pass None to restore the default implementation.
    """
    global _test_environment_checker
    with _test_environment_lock:
        _test_environment_checker = checker


def get_test_environment_checker() -> Optional[TestEnvironmentChecker]:
    """Get the current test environment checker function.

    Returns:
        The current checker function if set, None otherwise.
    """
    with _test_environment_lock:
        return _test_environment_checker


def _ensure_npm_global_in_path(
    env: dict[str, str],
    *,
    explicit_home: str | None = None,
) -> dict[str, str]:
    """Ensure common npm-global bin directories are in PATH.

    GUI-launched or launchd-spawned Python processes often inherit a minimal
    PATH that excludes user-level npm global installs (e.g. ~/.npm-global/bin).
    """
    home = os.path.expanduser("~") if explicit_home is None else explicit_home
    extra_dirs = []
    if home:
        extra_dirs.extend(
            (
                os.path.join(home, ".npm-global", "bin"),
                os.path.join(home, ".local", "bin"),
            )
        )
    extra_dirs.extend(("/opt/homebrew/bin", "/usr/local/bin"))
    current_path = env.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    path_set = set(path_parts)
    for d in extra_dirs:
        if d not in path_set and os.path.isdir(d):
            path_parts.append(d)
            path_set.add(d)
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def apply_anthropic_betas(env: dict[str, str], model_name: Optional[str]) -> dict[str, str]:
    """Merge ``ANTHROPIC_BETAS`` flags implied by *model_name* into *env*.

    Currently this is the 1M-context beta: when *model_name* carries the
    ``[1m]`` suffix that Claude Code CLI uses, ensure
    ``context-1m-2025-08-07`` is present in ``env['ANTHROPIC_BETAS']``
    (existing values are preserved and de-duplicated).

    Mutates *env* in place and returns it, mirroring
    :func:`_ensure_npm_global_in_path`.
    """
    name = (model_name or "").strip()
    if not name:
        return env

    # Late import keeps this leaf module free of intra-package cycles.
    from src.acp.claude_capabilities import CONTEXT_1M_BETA, is_1m_variant

    if not is_1m_variant(name):
        return env

    existing = (env.get("ANTHROPIC_BETAS") or "").strip()
    if not existing:
        env["ANTHROPIC_BETAS"] = CONTEXT_1M_BETA
        return env

    # De-dup while preserving original ordering.
    parts: list[str] = []
    seen: set[str] = set()
    for raw in existing.split(","):
        token = raw.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        parts.append(token)
    if CONTEXT_1M_BETA not in seen:
        parts.append(CONTEXT_1M_BETA)
    env["ANTHROPIC_BETAS"] = ",".join(parts)
    return env


def build_clean_env(base: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return a copy of *base* (default: ``os.environ``) with guard keys removed.

    This centralises the ``os.environ.copy(); env.pop("CLAUDECODE", None)``
    pattern used across ACP, agent-session, and provider modules.
    """
    env = dict(base) if base is not None else os.environ.copy()
    for key in _GUARD_KEYS:
        env.pop(key, None)
    get_uid = getattr(os, "getuid", None)
    user_scope = str(get_uid()) if callable(get_uid) else getpass.getuser()
    env.setdefault(
        "UV_CACHE_DIR",
        os.path.join(
            tempfile.gettempdir(),
            f"ghostap-uv-cache-{user_scope}",
        ),
    )
    env = _ensure_npm_global_in_path(
        env,
        explicit_home=None if base is None else env.get("HOME", ""),
    )
    return env


def _reset_env_for_testing() -> None:
    """Reset the env module's global state for testing.

    This function is only allowed in test environments.
    """
    import sys

    if "pytest" not in sys.modules:
        raise RuntimeError(
            "_reset_env_for_testing() is only allowed in test environments."
        )

    global _test_environment_checker
    with _test_environment_lock:
        _test_environment_checker = None
