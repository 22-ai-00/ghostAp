"""Regression contracts for the deny-by-default ingress review."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.access_control import (
    IngressAccessPolicy,
    IngressAccessPolicyProvider,
    IngressAccessRequest,
)
from src.admin_bootstrap import AdminBootstrapService
from src.config import IngressAccessMode
from src.config.env_file_store import AtomicEnvFileStore
from src.feishu.handlers.diagnostics import DiagnosticsHandler
from src.feishu.slash_command_parser import SlashCommandParser


def _policy(
    *,
    mode: IngressAccessMode = IngressAccessMode.ENFORCED,
    scope: str = "p2p_only",
) -> IngressAccessPolicy:
    return IngressAccessPolicy(
        admin_ids=frozenset(),
        allowed_user_ids=frozenset(),
        allowed_chat_ids=frozenset(),
        mode=mode,
        admin_bootstrap_scope=scope,
    )


def _request(**overrides: object) -> IngressAccessRequest:
    values: dict[str, object] = {
        "message_id": "om_valid",
        "sender_id": "ou_valid",
        "chat_id": "oc_valid",
        "chat_type": "p2p",
        "command_match": SlashCommandParser.parse("/setadmin"),
    }
    values.update(overrides)
    return IngressAccessRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("message_id", None),
        ("message_id", ""),
        ("message_id", "unknown"),
        ("message_id", "msg_legacy"),
        ("chat_id", None),
        ("chat_id", ""),
        ("chat_id", "unknown"),
        ("chat_id", "chat_legacy"),
        ("sender_id", None),
        ("sender_id", ""),
        ("sender_id", "unknown"),
        ("sender_id", "user_legacy"),
        ("chat_type", None),
        ("chat_type", ""),
        ("chat_type", "unknown"),
        ("chat_type", "P2P"),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        IngressAccessMode.ENFORCED,
        IngressAccessMode.SHADOW,
        IngressAccessMode.LEGACY_ALLOW_ALL,
    ],
)
def test_policy_rejects_noncanonical_request_facts_before_all_modes(
    field: str,
    invalid: object,
    mode: IngressAccessMode,
) -> None:
    decision = _policy(mode=mode, scope="any_chat").decide(
        _request(**{field: invalid})
    )

    assert decision.allowed is False
    assert decision.reason_code == "invalid_ingress_facts"


def _bootstrap_settings(*, scope: str = "p2p_only") -> SimpleNamespace:
    return SimpleNamespace(
        admin_user_ids=frozenset(),
        allowed_user_ids=frozenset(),
        allowed_chat_ids=frozenset(),
        ingress_access_mode="enforced",
        admin_bootstrap_scope=scope,
    )


@pytest.fixture(autouse=True)
def _clear_admin_rate_limit() -> None:
    AdminBootstrapService._last_attempt.clear()
    yield
    AdminBootstrapService._last_attempt.clear()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"message_id": ""},
        {"message_id": "unknown"},
        {"message_id": "msg_legacy"},
        {"sender_id": ""},
        {"sender_id": "unknown"},
        {"sender_id": "user_legacy"},
        {"chat_id": ""},
        {"chat_id": "unknown"},
        {"chat_id": "chat_legacy"},
        {"chat_type": ""},
        {"chat_type": "unknown"},
    ],
)
def test_bootstrap_service_rejects_malformed_request_context_without_writes(
    tmp_path,
    kwargs: dict[str, str],
) -> None:
    values = {
        "message_id": "om_valid",
        "sender_id": "ou_valid",
        "chat_id": "oc_valid",
        "chat_type": "p2p",
    }
    values.update(kwargs)
    store = MagicMock()
    result = AdminBootstrapService(
        env_path=tmp_path / ".env",
        settings_getter=_bootstrap_settings,
        env_store=store,
    ).set_admin(
        values["sender_id"],
        chat_type=values["chat_type"],
        chat_id=values["chat_id"],
        message_id=values["message_id"],
    )

    assert result.success is False
    assert result.code == "invalid_request_context"
    store.update_with.assert_not_called()
    store.update_many.assert_not_called()


def test_p2p_only_bootstrap_has_no_context_free_admin_backdoor(tmp_path) -> None:
    result = AdminBootstrapService(
        env_path=tmp_path / ".env",
        settings_getter=_bootstrap_settings,
    ).set_admin("ou_valid")

    assert result.success is False
    assert result.code == "invalid_request_context"
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("chat_type", ["p2p", "group"])
def test_any_chat_requires_a_real_supported_chat_context(
    tmp_path,
    chat_type: str,
) -> None:
    settings = _bootstrap_settings(scope="any_chat")
    result = AdminBootstrapService(
        env_path=tmp_path / ".env",
        settings_getter=lambda: settings,
    ).set_admin(
        "ou_valid",
        chat_type=chat_type,
        chat_id="oc_valid",
        message_id="om_valid",
    )

    assert result.success is True
    assert settings.allowed_chat_ids == frozenset({"oc_valid"})


def _make_ws_data() -> MagicMock:
    data = MagicMock()
    data.event.message.message_id = "om_valid"
    data.event.message.chat_id = "oc_valid"
    data.event.message.chat_type = "group"
    data.event.message.create_time = "9999999999999"
    data.event.message.message_type = "text"
    data.event.message.content = '{"text":"hello"}'
    data.event.message.parent_id = None
    data.event.message.root_id = None
    data.event.sender.sender_id.open_id = "ou_valid"
    data.event.sender.sender_id.union_id = "on_valid"
    return data


def _make_ws_client() -> MagicMock:
    from src.feishu.ws_client import FeishuWSClient

    client = FeishuWSClient.__new__(FeishuWSClient)
    client.settings = SimpleNamespace(
        admin_user_ids=frozenset(),
        allowed_user_ids=frozenset({"ou_valid"}),
        allowed_chat_ids=frozenset({"oc_valid"}),
        ingress_access_mode="enforced",
        admin_bootstrap_scope="p2p_only",
        thread_programming_enabled=False,
    )
    client._get_image_handler = MagicMock()
    parsed = MagicMock(text="hello", image_keys=[])
    client._get_image_handler.return_value.parse_message.return_value = parsed
    client._scheduler = MagicMock()
    client._message_linker = MagicMock()
    client._message_mapper = MagicMock()
    client._message_mapper.get_project_id.return_value = None
    client._project_manager = MagicMock()
    client._project_manager.get_active_project.return_value = None
    client._thread_manager = MagicMock()
    client._ensure_request_id = MagicMock(return_value="req_valid")
    client._pending_image_lock = MagicMock()
    client._pending_image_lock.__enter__ = MagicMock(return_value=None)
    client._pending_image_lock.__exit__ = MagicMock(return_value=False)
    client._pending_image_keys = {}
    client._pending_image_only = set()
    client._chat_lock_gate = MagicMock()
    client._employee_department_runtime = MagicMock()
    client._handle_image_content = MagicMock()
    client._dispatch_message_logic = MagicMock()
    client._dispatch_empty_text = MagicMock()
    client._get_api_client = MagicMock()
    client._validate_message = MagicMock(return_value=True)
    client._system_handler = MagicMock()
    return client


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        ("message.message_id", None),
        ("message.message_id", "unknown"),
        ("message.message_id", "msg_legacy"),
        ("message.chat_id", None),
        ("message.chat_id", "unknown"),
        ("message.chat_id", "chat_legacy"),
        ("message.chat_type", None),
        ("message.chat_type", "unknown"),
        ("sender.sender_id.open_id", None),
        ("sender.sender_id.open_id", "unknown"),
        ("sender.sender_id.open_id", "user_legacy"),
    ],
)
def test_ws_intake_rejects_malformed_facts_before_content_and_all_side_effects(
    path: str,
    invalid: object,
) -> None:
    client = _make_ws_client()
    data = _make_ws_data()
    target = data.event
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], invalid)

    client._handle_message(data)

    client._get_image_handler.assert_not_called()
    client._scheduler.submit.assert_not_called()
    client._message_linker.register_origin.assert_not_called()
    client._project_manager.get_active_project.assert_not_called()
    client._ensure_request_id.assert_not_called()


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        ("message.message_id", None),
        ("message.chat_id", "unknown"),
        ("message.chat_type", "GROUP"),
        ("sender.sender_id.open_id", "user_legacy"),
    ],
)
def test_ws_worker_rejects_malformed_current_facts_before_content_or_business(
    path: str,
    invalid: object,
) -> None:
    client = _make_ws_client()
    data = _make_ws_data()
    target = data.event
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], invalid)

    client._process_message_async(data)

    client._get_image_handler.assert_not_called()
    client._validate_message.assert_not_called()
    client._chat_lock_gate.check.assert_not_called()
    client._employee_department_runtime.record_group_event.assert_not_called()
    client._handle_image_content.assert_not_called()
    client._dispatch_message_logic.assert_not_called()


def test_worker_redecides_from_mutated_event_facts_instead_of_task_spec() -> None:
    client = _make_ws_client()
    data = _make_ws_data()
    task_ctx = SimpleNamespace(
        spec=SimpleNamespace(
            sender_id="ou_valid",
            is_p2p=False,
            sender_union_id="on_valid",
            tenant_key="tenant",
        )
    )
    data.event.sender.sender_id.open_id = "ou_denied"

    client._process_message_async(data, task_ctx=task_ctx)

    client._get_image_handler.assert_called_once()
    client._validate_message.assert_not_called()
    client._dispatch_message_logic.assert_not_called()


def test_env_store_distinguishes_pre_replace_failure(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ADMIN_USER_IDS=ou_old\n", encoding="utf-8")
    store = AtomicEnvFileStore(env_path)

    with patch("src.config.env_file_store.os.replace", side_effect=OSError("full")):
        with pytest.raises(OSError) as exc_info:
            store.update_many({"ADMIN_USER_IDS": "ou_new"})

    assert type(exc_info.value).__name__ == "EnvPreReplaceError"
    assert env_path.read_text(encoding="utf-8") == "ADMIN_USER_IDS=ou_old\n"


def test_env_store_reconciles_actual_file_after_post_replace_failure(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ADMIN_USER_IDS=ou_old\n", encoding="utf-8")
    store = AtomicEnvFileStore(env_path)
    real_fsync = os.fsync
    call_count = 0

    def fail_directory_fsync(fd: int) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("directory fsync failed")
        real_fsync(fd)

    with patch("src.config.env_file_store.os.fsync", side_effect=fail_directory_fsync):
        with pytest.raises(OSError) as exc_info:
            store.update_many({"ADMIN_USER_IDS": "ou_new"})

    assert type(exc_info.value).__name__ == "EnvCommitUncertainError"
    snapshot = exc_info.value.snapshot
    assert snapshot.values["ADMIN_USER_IDS"] == "ou_new"
    assert env_path.read_text(encoding="utf-8") == "ADMIN_USER_IDS=ou_new\n"


def test_env_store_does_not_chmod_after_atomic_replace(tmp_path) -> None:
    env_path = tmp_path / ".env"
    store = AtomicEnvFileStore(env_path)

    with patch("src.config.env_file_store.os.chmod") as chmod:
        store.update_many({"ADMIN_USER_IDS": "ou_new"})

    chmod.assert_not_called()
    assert os.stat(env_path).st_mode & 0o777 == 0o600


def test_env_store_lock_close_failure_preserves_confirmed_commit_snapshot(
    tmp_path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ADMIN_USER_IDS=ou_old\n", encoding="utf-8")
    store = AtomicEnvFileStore(env_path)
    real_close = os.close
    call_count = 0

    def fail_lock_close(fd: int) -> None:
        nonlocal call_count
        call_count += 1
        real_close(fd)
        if call_count == 2:
            raise OSError("lock close failed")

    with patch("src.config.env_file_store.os.close", side_effect=fail_lock_close):
        with pytest.raises(OSError) as exc_info:
            store.update_many({"ADMIN_USER_IDS": "ou_new"})

    assert type(exc_info.value).__name__ == "EnvPostCommitCleanupError"
    assert exc_info.value.snapshot.values["ADMIN_USER_IDS"] == "ou_new"
    assert env_path.read_text(encoding="utf-8") == "ADMIN_USER_IDS=ou_new\n"


def test_env_store_parent_close_after_successful_fsync_is_cleanup_failure(
    tmp_path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ADMIN_USER_IDS=ou_old\n", encoding="utf-8")
    store = AtomicEnvFileStore(env_path)
    real_close = os.close
    call_count = 0

    def fail_parent_close(fd: int) -> None:
        nonlocal call_count
        call_count += 1
        real_close(fd)
        if call_count == 1:
            raise OSError("parent close failed")

    with patch("src.config.env_file_store.os.close", side_effect=fail_parent_close):
        with pytest.raises(OSError) as exc_info:
            store.update_many({"ADMIN_USER_IDS": "ou_new"})

    assert type(exc_info.value).__name__ == "EnvPostCommitCleanupError"
    assert exc_info.value.snapshot.values["ADMIN_USER_IDS"] == "ou_new"
    assert env_path.read_text(encoding="utf-8") == "ADMIN_USER_IDS=ou_new\n"


def test_allow_chat_reads_quoted_export_dotenv_with_project_semantics(
    tmp_path,
) -> None:
    settings = SimpleNamespace(
        admin_user_ids=frozenset({"ou_admin"}),
        allowed_user_ids=frozenset({"ou_admin"}),
        allowed_chat_ids=frozenset({"oc_dm"}),
        ingress_access_mode="enforced",
        admin_bootstrap_scope="p2p_only",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        'export ADMIN_USER_IDS = "ou_admin"\n'
        "ALLOWED_USER_IDS='ou_admin'\n"
        'export ALLOWED_CHAT_IDS = "oc_dm"\n',
        encoding="utf-8",
    )

    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
    ).allow_current_chat(
        "ou_admin",
        "oc_group",
        chat_type="group",
        message_id="om_allow_quoted",
    )

    assert result.success is True
    assert settings.allowed_chat_ids == frozenset({"oc_dm", "oc_group"})
    assert "ALLOWED_CHAT_IDS=oc_dm,oc_group\n" in env_path.read_text(
        encoding="utf-8"
    )


def test_service_reconciles_live_policy_and_blocks_on_commit_uncertain(
    tmp_path,
) -> None:
    settings = _bootstrap_settings()
    provider = IngressAccessPolicyProvider(_policy())
    store = AtomicEnvFileStore(tmp_path / ".env")
    with patch.object(
        store,
        "_fsync_parent",
        side_effect=OSError("directory fsync failed"),
    ):
        result = AdminBootstrapService(
            env_path=tmp_path / ".env",
            settings_getter=lambda: settings,
            policy_provider=provider,
            env_store=store,
        ).set_admin(
            "ou_valid",
            chat_type="p2p",
            chat_id="oc_valid",
            message_id="om_valid",
        )

    assert result.success is False
    assert result.code == "commit_uncertain"
    assert provider.current.admin_ids == frozenset({"ou_valid"})
    assert provider.current.allowed_user_ids == frozenset({"ou_valid"})
    assert provider.current.allowed_chat_ids == frozenset({"oc_valid"})
    assert settings.admin_user_ids == frozenset({"ou_valid"})
    finding = next(
        item
        for item in provider.blocking_findings
        if item.code == "ingress_env_commit_uncertain"
    )
    assert finding.severity.value == "blocking"


def test_service_publishes_confirmed_commit_and_blocks_on_lock_cleanup_failure(
    tmp_path,
) -> None:
    settings = _bootstrap_settings()
    provider = IngressAccessPolicyProvider(_policy())
    real_close = os.close
    call_count = 0

    def fail_lock_close(fd: int) -> None:
        nonlocal call_count
        call_count += 1
        real_close(fd)
        if call_count == 2:
            raise OSError("lock close failed")

    with patch("src.config.env_file_store.os.close", side_effect=fail_lock_close):
        result = AdminBootstrapService(
            env_path=tmp_path / ".env",
            settings_getter=lambda: settings,
            policy_provider=provider,
        ).set_admin(
            "ou_valid",
            chat_type="p2p",
            chat_id="oc_valid",
            message_id="om_valid",
        )

    assert result.success is False
    assert result.code == "commit_cleanup_failed"
    assert provider.current.admin_ids == frozenset({"ou_valid"})
    assert settings.admin_user_ids == frozenset({"ou_valid"})
    assert any(
        item.code == "ingress_env_post_commit_cleanup_failed"
        for item in provider.blocking_findings
    )


def test_settings_mirror_failure_rolls_back_and_blocks_runtime_publication(
    tmp_path,
) -> None:
    class PartiallyReadOnlySettings:
        def __init__(self) -> None:
            self.admin_user_ids = frozenset()
            self._allowed_user_ids = frozenset()
            self.allowed_chat_ids = frozenset()
            self.ingress_access_mode = "enforced"
            self.admin_bootstrap_scope = "p2p_only"

        @property
        def allowed_user_ids(self) -> frozenset[str]:
            return self._allowed_user_ids

        @allowed_user_ids.setter
        def allowed_user_ids(self, _value: frozenset[str]) -> None:
            raise AttributeError("settings mirror is read-only")

    settings = PartiallyReadOnlySettings()
    provider = IngressAccessPolicyProvider(_policy())
    original = provider.current

    result = AdminBootstrapService(
        env_path=tmp_path / ".env",
        settings_getter=lambda: settings,
        policy_provider=provider,
    ).set_admin(
        "ou_valid",
        chat_type="p2p",
        chat_id="oc_valid",
        message_id="om_valid",
    )

    assert result.success is False
    assert result.code == "settings_mirror_failed"
    assert provider.current is original
    assert settings.admin_user_ids == frozenset()
    assert settings.allowed_user_ids == frozenset()
    assert settings.allowed_chat_ids == frozenset()
    finding = next(
        item
        for item in provider.blocking_findings
        if item.code == "ingress_settings_mirror_failed"
    )
    assert finding.severity.value == "blocking"
    assert "ADMIN_USER_IDS=ou_valid\n" in (tmp_path / ".env").read_text(
        encoding="utf-8"
    )


def _allow_chat_in_process(env_path: str, chat_id: str) -> None:
    settings = SimpleNamespace(
        admin_user_ids=frozenset({"ou_admin"}),
        allowed_user_ids=frozenset({"ou_admin"}),
        allowed_chat_ids=frozenset({"oc_dm"}),
        ingress_access_mode="enforced",
        admin_bootstrap_scope="p2p_only",
    )
    result = AdminBootstrapService(
        env_path=env_path,
        settings_getter=lambda: settings,
    ).allow_current_chat(
        "ou_admin",
        chat_id,
        chat_type="group",
        message_id=f"om_{chat_id[3:]}",
    )
    if not result.success:
        raise RuntimeError(result.code)


def test_allow_chat_processes_merge_inside_flock_without_lost_updates(
    tmp_path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ADMIN_USER_IDS=ou_admin\n"
        "ALLOWED_USER_IDS=ou_admin\n"
        "ALLOWED_CHAT_IDS=oc_dm\n",
        encoding="utf-8",
    )
    chats = [f"oc_group_{index}" for index in range(8)]
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_allow_chat_in_process, args=(str(env_path), chat))
        for chat in chats
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    line = next(
        item
        for item in env_path.read_text(encoding="utf-8").splitlines()
        if item.startswith("ALLOWED_CHAT_IDS=")
    )
    assert set(line.partition("=")[2].split(",")) == {"oc_dm", *chats}


def test_admin_audit_never_contains_raw_sender_or_target(tmp_path) -> None:
    settings = _bootstrap_settings()
    with patch("src.admin_bootstrap.audit_logger") as audit:
        result = AdminBootstrapService(
            env_path=tmp_path / ".env",
            settings_getter=lambda: settings,
        ).set_admin(
            "ou_sensitive_sender",
            chat_type="p2p",
            chat_id="oc_sensitive_chat",
            message_id="om_sensitive_message",
        )

    assert result.success is True
    rendered = " ".join(str(item) for item in audit.info.call_args.args)
    assert "ou_sensitive_sender" not in rendered
    assert hashlib.sha256(b"ou_sensitive_sender").hexdigest()[:16] in rendered


@pytest.mark.parametrize(
    ("mode", "scope", "expected_code"),
    [
        ("shadow", "p2p_only", "ingress_shadow_not_enforcing"),
        ("legacy_allow_all", "p2p_only", "ingress_legacy_allow_all"),
        ("enforced", "any_chat", "admin_bootstrap_any_chat"),
    ],
)
def test_status_surfaces_security_posture_warnings(
    mode: str,
    scope: str,
    expected_code: str,
) -> None:
    settings = SimpleNamespace(
        ingress_access_mode=mode,
        admin_bootstrap_scope=scope,
        shell_security_profile="restricted",
        shell_high_risk_confirmation=True,
        shell_blocked_patterns=(),
        admin_user_ids=frozenset(),
        allowed_user_ids=frozenset(),
        allowed_chat_ids=frozenset(),
    )
    ctx = MagicMock()
    ctx.settings = settings
    ctx.ingress_access_policy_provider = IngressAccessPolicyProvider(
        IngressAccessPolicy(
            admin_ids=frozenset(),
            allowed_user_ids=frozenset(),
            allowed_chat_ids=frozenset(),
            mode=IngressAccessMode(mode),
            admin_bootstrap_scope=scope,
        )
    )
    handler = DiagnosticsHandler(ctx)
    handler.reply_card = MagicMock()
    handler.get_working_dir = MagicMock(return_value="/tmp")
    handler._build_lock_status_lines = MagicMock(return_value="")

    with (
        patch(
            "src.feishu.handlers.diagnostics.DiagnosticsHelper.get_all_engine_statuses",
            return_value=[],
        ),
        patch(
            "src.feishu.handlers.diagnostics.CardBuilder.build_unified_status_content",
            return_value="base",
        ),
        patch(
            "src.feishu.handlers.diagnostics.CardBuilder.build_smart_response_card",
            return_value=("interactive", {"card": "ok"}),
        ) as build_card,
    ):
        handler.show_unified_status("om_status", "oc_status", "/status")

    assert expected_code in build_card.call_args.kwargs["content"]


def test_status_surfaces_shared_provider_blocking_findings() -> None:
    settings = SimpleNamespace(
        ingress_access_mode="enforced",
        admin_bootstrap_scope="p2p_only",
        shell_security_profile="restricted",
        shell_high_risk_confirmation=True,
        shell_blocked_patterns=(),
        admin_user_ids=frozenset(),
        allowed_user_ids=frozenset(),
        allowed_chat_ids=frozenset(),
    )
    provider = IngressAccessPolicyProvider(_policy())
    provider.record_blocking_finding(
        "ingress_env_commit_uncertain",
        "dotenv replacement durability is uncertain",
    )
    ctx = MagicMock(settings=settings)
    ctx.settings = settings
    ctx.ingress_access_policy_provider = provider
    handler = DiagnosticsHandler(ctx)
    handler.reply_card = MagicMock()
    handler.get_working_dir = MagicMock(return_value="/tmp")
    handler._build_lock_status_lines = MagicMock(return_value="")

    with (
        patch(
            "src.feishu.handlers.diagnostics.DiagnosticsHelper.get_all_engine_statuses",
            return_value=[],
        ),
        patch(
            "src.feishu.handlers.diagnostics.CardBuilder.build_unified_status_content",
            return_value="base",
        ),
        patch(
            "src.feishu.handlers.diagnostics.CardBuilder.build_smart_response_card",
            return_value=("interactive", {"card": "ok"}),
        ) as build_card,
    ):
        handler.show_unified_status("om_status", "oc_status", "/status")

    content = build_card.call_args.kwargs["content"]
    assert "ingress_env_commit_uncertain" in content
    assert "BLOCKING" in content
