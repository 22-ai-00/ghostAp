"""Regression tests for shell execution while another task owns the repo."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.lock_helper import LockHelper
from src.feishu.handlers.system import SystemHandler
from src.repo_lock import LockConflictError, RepoLockManager
from src.tasking import TaskScheduler, TaskSpec
from src.thread import set_current_is_p2p
from src.utils.signing import VerifyResult, verify_command_sig


class _LockHandler:
    def __init__(self, manager: RepoLockManager) -> None:
        self.ctx = SimpleNamespace(repo_lock_manager=manager)
        self.settings = SimpleNamespace(repo_lock_hard_timeout=3600)


def test_strict_repo_lock_rejects_second_p2p_chat(tmp_path) -> None:
    """P2P privilege must not let shell mutate a repo owned by another chat."""
    manager = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
    helper = LockHelper(_LockHandler(manager))
    body = MagicMock()
    root_path = str(tmp_path)
    manager.acquire(root_path, "active-task-chat")
    set_current_is_p2p(True)
    try:
        with pytest.raises(LockConflictError):
            helper._with_repo_lock_strict(root_path, "restart-chat", body)
        body.assert_not_called()
    finally:
        set_current_is_p2p(False)
        manager.shutdown()


def test_strict_shell_same_chat_cannot_overlap_programming_run(tmp_path) -> None:
    """Different scheduler runs in one chat must not be lock-reentrant."""
    manager = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
    helper = LockHelper(_LockHandler(manager))
    scheduler = TaskScheduler(max_concurrent=2, per_key_concurrency=1)
    programming_started = threading.Event()
    programming_finished = threading.Event()
    release_programming = threading.Event()
    shell_finished = threading.Event()
    shell_body = MagicMock()

    def programming(_ctx):
        def body():
            programming_started.set()
            assert release_programming.wait(timeout=2)

        try:
            return helper._with_repo_lock_strict(str(tmp_path), "same-chat", body)
        finally:
            programming_finished.set()

    scheduler.submit(
        TaskSpec(
            chat_id="same-chat",
            name="programming",
            queue_key="same-chat:project",
        ),
        programming,
    )
    assert programming_started.wait(timeout=1)

    def shell(_ctx):
        try:
            with pytest.raises(LockConflictError):
                helper._with_repo_lock_strict(
                    str(tmp_path),
                    "same-chat",
                    shell_body,
                )
            return "blocked"
        finally:
            shell_finished.set()

    scheduler.submit(
        TaskSpec(
            chat_id="same-chat",
            name="shell-fast",
            is_system_command=True,
        ),
        shell,
    )

    try:
        assert shell_finished.wait(timeout=2)
        shell_body.assert_not_called()
    finally:
        release_programming.set()
        assert programming_finished.wait(timeout=2)
        scheduler.stop(wait=True, shutdown_executor=True)
        manager.shutdown()


@pytest.mark.slow
def test_short_repo_lease_is_renewed_while_strict_task_is_active(
    tmp_path,
) -> None:
    manager = RepoLockManager(
        idle_timeout=300,
        cleanup_interval=0.02,
        hard_timeout=0.6,
    )
    helper = LockHelper(_LockHandler(manager))
    lock_still_held: list[bool] = []

    def active_body() -> None:
        time.sleep(1.1)
        lock_still_held.append(manager.get_lock_info(str(tmp_path)) is not None)

    try:
        helper._with_repo_lock_strict(
            str(tmp_path),
            "short-lease-chat",
            active_body,
        )
        assert lock_still_held == [True]
    finally:
        manager.shutdown()


def test_system_shell_uses_strict_repo_lock() -> None:
    """The shell entry point must call the strict helper, not P2P-bypass mode."""
    ctx = MagicMock()
    ctx.main_bot_outbound_audit = None
    ctx.main_bot_outbound_audit_failure = None
    ctx.tenant_key_resolver = None
    handler = SystemHandler(ctx)
    expected = object()
    handler.lock_helper._with_repo_lock = MagicMock(
        side_effect=AssertionError("bypass-capable lock helper was used")
    )
    handler.lock_helper._with_repo_lock_strict = MagicMock(return_value=expected)

    result = handler.execute_shell_and_reply(
        "message-1",
        "restart-chat",
        "./restart.sh rr",
        "/repo",
    )

    assert result is expected
    handler.lock_helper._with_repo_lock_strict.assert_called_once()


def test_projectless_shell_nested_git_cwd_uses_repo_root_lock_key(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    nested_cwd = repo_root / "src" / "feature"
    nested_cwd.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    ctx = MagicMock()
    ctx.main_bot_outbound_audit = None
    ctx.main_bot_outbound_audit_failure = None
    ctx.tenant_key_resolver = None
    handler = SystemHandler(ctx)
    handler.lock_helper._with_repo_lock_strict = MagicMock(return_value=object())

    handler.execute_shell_and_reply(
        "message-1",
        "shell-chat",
        "pwd",
        str(nested_cwd),
    )

    lock_path = handler.lock_helper._with_repo_lock_strict.call_args.args[0]
    assert lock_path == str(repo_root.resolve())


@pytest.mark.parametrize(
    "git_failure",
    [
        subprocess.CompletedProcess(
            args=["git", "rev-parse"],
            returncode=128,
            stdout="",
            stderr="fatal",
        ),
        subprocess.TimeoutExpired(cmd=["git", "rev-parse"], timeout=2),
        OSError("git unavailable"),
    ],
    ids=["nonzero", "timeout", "oserror"],
)
def test_nested_git_cwd_falls_back_to_worktree_root_when_git_probe_fails(
    tmp_path,
    git_failure,
) -> None:
    """A failed Git probe must not split one worktree into per-cwd lock keys."""
    repo_root = tmp_path / "repo"
    nested_cwd = repo_root / "src" / "feature"
    nested_cwd.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )

    manager = RepoLockManager(idle_timeout=300, cleanup_interval=9999)
    helper = LockHelper(_LockHandler(manager))
    manager.acquire(str(repo_root.resolve()), "active-task-chat")
    try:
        if isinstance(git_failure, subprocess.CompletedProcess):
            probe = patch(
                "src.feishu.handlers.lock_helper.subprocess.run",
                return_value=git_failure,
            )
        else:
            probe = patch(
                "src.feishu.handlers.lock_helper.subprocess.run",
                side_effect=git_failure,
            )

        with probe:
            lock_root = helper.resolve_git_lock_root(str(nested_cwd))

        assert lock_root == str(repo_root.resolve())
        with pytest.raises(LockConflictError):
            helper._with_repo_lock_strict(
                lock_root,
                "shell-chat",
                MagicMock(),
            )
    finally:
        manager.shutdown()


def test_projectless_shell_non_git_cwd_keeps_existing_lock_key(tmp_path) -> None:
    working_dir = tmp_path / "plain" / "nested"
    working_dir.mkdir(parents=True)
    ctx = MagicMock()
    ctx.main_bot_outbound_audit = None
    ctx.main_bot_outbound_audit_failure = None
    ctx.tenant_key_resolver = None
    handler = SystemHandler(ctx)
    handler.lock_helper._with_repo_lock_strict = MagicMock(return_value=object())

    handler.execute_shell_and_reply(
        "message-1",
        "shell-chat",
        "pwd",
        str(working_dir),
    )

    lock_path = handler.lock_helper._with_repo_lock_strict.call_args.args[0]
    assert lock_path == str(working_dir)


def _find_retry_action_value(card_json: str) -> dict:
    pending = [json.loads(card_json)]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            value = node.get("value")
            if isinstance(value, dict) and value.get("action") == "retry_command":
                return value
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    raise AssertionError("retry_command action not found")


def test_strict_shell_conflict_retry_is_chat_bound_and_single_use(
    tmp_path,
) -> None:
    """Strict-shell conflict cards must use expiring, anti-replay v2 signatures."""
    import src.utils.signing as signing

    ctx = MagicMock()
    ctx.main_bot_outbound_audit = None
    ctx.main_bot_outbound_audit_failure = None
    ctx.tenant_key_resolver = None
    ctx.chat_lock_manager = None
    ctx.repo_lock_manager = None
    ctx.settings.repo_lock_idle_timeout = 300
    handler = SystemHandler(ctx)
    handler.reply_card = MagicMock()
    handler.reply_text = MagicMock()
    conflict = LockConflictError(
        "busy",
        holder_chat_id="other-chat",
        locked_since=1.0,
        root_path="/repo",
        last_active_time=1.0,
    )
    handler.lock_helper._with_repo_lock_strict = MagicMock(side_effect=conflict)
    signing._USED_NONCES.clear()

    with (
        patch(
            "src.utils.signing._get_signing_key",
            return_value="strict-shell-test-secret",
        ),
        patch(
            "src.utils.signing._nonce_store_path",
            return_value=tmp_path / "used-command-nonces.json",
        ),
        patch(
            "src.config.get_settings",
            return_value=SimpleNamespace(app_id=""),
        ),
    ):
        handler.execute_shell_and_reply(
            "message-1",
            "shell-chat",
            "./restart.sh rr",
            "/repo",
        )
        action = _find_retry_action_value(handler.reply_card.call_args.args[1])
        signature = action["_s"]

        assert verify_command_sig(
            action["_t"],
            signature,
            chat_id="other-chat",
        ) is VerifyResult.CHAT_MISMATCH
        assert verify_command_sig(
            action["_t"],
            signature,
            chat_id="shell-chat",
        ) is VerifyResult.OK
        assert verify_command_sig(
            action["_t"],
            signature,
            chat_id="shell-chat",
        ) is VerifyResult.NONCE_REUSED
