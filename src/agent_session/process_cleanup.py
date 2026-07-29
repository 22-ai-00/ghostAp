"""Bounded subprocess-tree cleanup for per-prompt CLI backends."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)


def _wait_for_leader(
    process: subprocess.Popen,
    *,
    timeout: float,
    label: str,
) -> bool:
    try:
        process.wait(timeout=max(0.0, timeout))
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        logger.debug("[%s] subprocess wait failed", label, exc_info=True)
        return False


def _terminate_direct_process(
    process: subprocess.Popen,
    *,
    terminate_grace: float,
    kill_grace: float,
    label: str,
) -> bool:
    """Fallback cleanup where POSIX process groups are unavailable."""
    try:
        if process.poll() is not None:
            return True
    except Exception:
        return False

    try:
        process.terminate()
    except Exception:
        logger.debug("[%s] subprocess terminate failed", label, exc_info=True)
    if _wait_for_leader(
        process,
        timeout=terminate_grace,
        label=label,
    ):
        return True

    try:
        process.kill()
    except Exception:
        logger.error("[%s] subprocess kill failed", label, exc_info=True)
        return False
    if _wait_for_leader(process, timeout=kill_grace, label=label):
        return True
    logger.error("[%s] subprocess did not exit after SIGKILL", label)
    return False


def _wait_for_process_group_exit(
    process_group_id: int,
    *,
    deadline: float,
    label: str,
) -> bool:
    """Wait until no process remains in a group, within one shared deadline."""
    while True:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            logger.error(
                "[%s] failed to probe subprocess group %s",
                label,
                process_group_id,
                exc_info=True,
            )
            return False

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def terminate_and_reap_process_tree(
    process: subprocess.Popen,
    *,
    process_group_id: int | None,
    terminate_grace: float,
    kill_grace: float,
    label: str,
) -> bool:
    """Stop one CLI process tree with bounded TERM→KILL escalation.

    ``process_group_id`` must only be supplied for a process launched with
    ``start_new_session=True``.  The group is probed even after its leader has
    exited because a descendant may still own stdout/stderr and keep a reader
    blocked indefinitely.
    """
    if (
        os.name != "posix"
        or not isinstance(process_group_id, int)
        or process_group_id <= 0
    ):
        return _terminate_direct_process(
            process,
            terminate_grace=terminate_grace,
            kill_grace=kill_grace,
            label=label,
        )

    group_id = process_group_id
    terminate_deadline = time.monotonic() + max(
        0.0,
        terminate_grace,
    )
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        # No descendant remains. Reap the leader if its status has not yet
        # been collected.
        try:
            if process.poll() is not None:
                return True
        except Exception:
            return False
        return _wait_for_leader(
            process,
            timeout=max(0.0, terminate_deadline - time.monotonic()),
            label=label,
        )
    except (PermissionError, OSError):
        logger.error(
            "[%s] failed to signal subprocess group %s",
            label,
            group_id,
            exc_info=True,
        )
        # Reap the direct process where possible, but report failure because
        # descendant termination was not confirmed.
        _terminate_direct_process(
            process,
            terminate_grace=terminate_grace,
            kill_grace=kill_grace,
            label=label,
        )
        return False

    leader_reaped = _wait_for_leader(
        process,
        timeout=max(0.0, terminate_deadline - time.monotonic()),
        label=label,
    )

    # The leader may exit on TERM while a descendant ignores it. Never use
    # leader reaping alone as proof that the stdout-owning process tree ended.
    if _wait_for_process_group_exit(
        group_id,
        deadline=terminate_deadline,
        label=label,
    ):
        return leader_reaped

    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        return leader_reaped
    except (PermissionError, OSError):
        logger.error(
            "[%s] failed to kill subprocess group %s",
            label,
            group_id,
            exc_info=True,
        )
        return False

    if not leader_reaped:
        leader_reaped = _wait_for_leader(
            process,
            timeout=kill_grace,
            label=label,
        )
    if not leader_reaped:
        logger.error("[%s] subprocess leader was not reaped", label)
        return False

    # SIGKILL cannot be handled or ignored. Descendants may briefly remain as
    # zombies until their new parent reaps them, but no live process can retain
    # the inherited pipes after a successful group signal.
    return True
