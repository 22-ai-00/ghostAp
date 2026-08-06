"""Path utilities (pure stdlib helpers).

本模块必须保持“纯库”属性：只依赖标准库（pathlib/typing 等），
避免引入 ACP/Feishu 等业务依赖，以杜绝循环依赖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def canonicalize_user_home_path(path: str | Path) -> Path:
    """Resolve only a system-managed user-home symlink in ``path``.

    Security-sensitive stores walk their own roots with ``O_NOFOLLOW``.  Some
    supported hosts expose ``Path.home()`` through a symlink (for example,
    ``/home/user`` -> ``/data/home/user``), so that trusted prefix must be
    canonicalized before the walk.  Child components are deliberately left
    unresolved so each store can continue rejecting links inside its root.
    """

    expanded = Path(os.path.abspath(Path(path).expanduser()))
    home = Path(os.path.abspath(Path.home()))
    try:
        relative = expanded.relative_to(home)
    except ValueError:
        return expanded
    resolved_home = home.resolve(strict=True)
    if not resolved_home.is_dir():
        raise ValueError("user home is not a directory")
    return resolved_home / relative


def normalize_session_cwd(cwd: Optional[str]) -> Optional[str]:
    """Return a stable absolute working directory for session startup.

    This is best-effort and deliberately does not resolve symlinks: callers
    keep the path spelling they supplied while relative paths stop depending
    on later process-wide working-directory changes.
    """
    raw = (cwd or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path.absolute())
    except Exception:
        return None


def normalize_repo_path(path: Optional[str]) -> Optional[str]:
    """将仓库路径归一化为唯一的绝对路径（用于仓库锁 key）。

    - None/空串 -> None
    - 其他 -> ``os.path.realpath(os.path.expanduser(path))``

    这里会展开符号链接（realpath），确保 ``~/repo``、``/home/user/repo``、
    ``/home/user/./repo`` 等价路径
    归一化后产出相同字符串，用于仓库级互斥锁的 key 比较。
    """
    raw = (path or "").strip()
    if not raw:
        return None
    try:
        return os.path.realpath(os.path.expanduser(raw))
    except Exception:
        return None
