"""Cross-process lease for one Spec checkpoint recovery attempt."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path


class SpecRecoveryLease:
    """Own an exclusive advisory lock until recovery reaches a terminal edge."""

    __slots__ = ("_fd",)

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "SpecRecoveryLease":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


def try_acquire_recovery_lease(
    checkpoint_path: str | os.PathLike[str],
) -> SpecRecoveryLease | None:
    """Acquire the checkpoint's lease without waiting for another process."""

    state_path = Path(checkpoint_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(f".{state_path.name}.recovery.lock")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return None
        raise
    return SpecRecoveryLease(fd)


__all__ = ["SpecRecoveryLease", "try_acquire_recovery_lease"]
