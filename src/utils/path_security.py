"""Path security utilities for employee engine.

Provides path restriction and blacklist checking utilities shared between
ACP client and MemoryManager to ensure consistent security policies.
"""

from __future__ import annotations

import logging
import os

from ..utils.errors import get_error_detail

logger = logging.getLogger(__name__)


def get_tool_path_restrictions() -> list[str]:
    """Read employee_tool_path_restrictions from Settings (lazy, best-effort)."""
    try:
        from ..config import get_settings
        return getattr(get_settings(), "employee_tool_path_restrictions", []) or []
    except Exception:
        return []


def check_path_restriction(path: str, restrictions: list[str]) -> bool:
    """Check if *path* is allowed by the restrictions whitelist.

    Returns:
        True  – if restrictions is empty (allow-all, backward compatible)
        True  – if the normalized path starts with any of the normalized restriction prefixes
        False – otherwise (path not in whitelist)
    """
    if not restrictions:
        return True
    # Use abspath for non-existent paths; realpath only when path exists (resolves symlinks)
    abs_path = os.path.normpath(os.path.abspath(path))
    norm_path = os.path.normpath(os.path.realpath(path)) if os.path.exists(path) else abs_path
    for prefix in restrictions:
        if not prefix:
            continue
        abs_prefix = os.path.normpath(os.path.abspath(prefix))
        norm_prefix = os.path.normpath(os.path.realpath(prefix)) if os.path.exists(prefix) else abs_prefix
        # Check both abspath and realpath variants
        if norm_path == norm_prefix or norm_path.startswith(norm_prefix + os.sep):
            return True
        if abs_path == abs_prefix or abs_path.startswith(abs_prefix + os.sep):
            return True
    return False


def get_acp_blacklist() -> tuple[list[str], list[str], list[str]]:
    """Read ACP blacklist settings (lazy, best-effort).

    Returns:
        Tuple of (blacklist_files, blacklist_dirs, blacklist_exts)
    """
    try:
        from ..config import get_settings
        settings = get_settings()
        files = getattr(settings, "employee_acp_blacklist_files", []) or []
        dirs = getattr(settings, "employee_acp_blacklist_dirs", []) or []
        exts = getattr(settings, "employee_acp_blacklist_exts", []) or []
        return files, dirs, exts
    except Exception:
        return [], [], []


def is_path_blacklisted(path: str) -> bool:
    """Check if *path* matches any blacklist rule.

    Canonical implementation used by both ACP client and MemoryManager.
    Uses fail-closed semantics: returns True (deny) on exceptions.

    Args:
        path: The file path to check.

    Returns:
        True if path is blacklisted, False otherwise.
    """
    if not path:
        return False

    blacklist_files, blacklist_dirs, blacklist_exts = get_acp_blacklist()

    # All blacklists empty: nothing is blacklisted
    if not blacklist_files and not blacklist_dirs and not blacklist_exts:
        return False

    try:
        # Normalize path (same approach as check_path_restriction)
        abs_path = os.path.normpath(os.path.abspath(path))
        norm_path = os.path.normpath(os.path.realpath(path)) if os.path.exists(path) else abs_path

        # Check both normalized paths
        for check_path in (abs_path, norm_path):
            # Check filename
            filename = os.path.basename(check_path)
            if filename and filename in blacklist_files:
                return True

            # Check directory patterns (path contains blacklisted directory)
            for blacklist_dir in blacklist_dirs:
                if not blacklist_dir:
                    continue
                # Normalize the blacklist directory pattern
                norm_blacklist_dir = blacklist_dir.rstrip(os.sep)
                # Check if path contains the blacklisted directory
                # e.g., ".ssh/" should match "/home/user/.ssh/id_rsa"
                dir_sep = os.sep + norm_blacklist_dir + os.sep
                if dir_sep in check_path + os.sep:
                    return True
                # Also check if path starts with the blacklisted directory
                if check_path.startswith(norm_blacklist_dir + os.sep):
                    return True

            # Check file extension
            _, ext = os.path.splitext(check_path)
            if ext and ext in blacklist_exts:
                return True

        return False
    except Exception as e:
        error_detail = get_error_detail(e)
        logger.error(
            "[PathSecurity] is_path_blacklisted error (fail-closed): path=%s error=%s",
            path, error_detail,
        )
        return True


def is_path_within_base(path: str, base_path: str) -> bool:
    """Check if *path* is within *base_path* (prevents path traversal).

    Returns:
        True if path is within base_path, False otherwise.
    """
    abs_base = os.path.normpath(os.path.abspath(base_path))
    abs_path = os.path.normpath(os.path.abspath(path))
    # Check both abspath and realpath (for symlink resolution)
    real_base = os.path.normpath(os.path.realpath(base_path)) if os.path.exists(base_path) else abs_base
    real_path = os.path.normpath(os.path.realpath(path)) if os.path.exists(path) else abs_path

    return (
        (abs_path == abs_base or abs_path.startswith(abs_base + os.sep))
        or (real_path == real_base or real_path.startswith(real_base + os.sep))
    )
