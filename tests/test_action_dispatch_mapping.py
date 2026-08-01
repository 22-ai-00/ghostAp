"""Tests for action_dispatch registries — ensure every worktree action_id constant
has a matching factory in build_worktree_action_registry() and each factory returns a CardEvent.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.card.actions import dispatch as action_ids
from src.card.actions.dispatch import build_worktree_action_registry
from src.card.error_diagnostics import error_diagnostic_store
from src.card.events import CardEvent


class _RegistryCaptureClient:
    """Minimal client double that captures Feishu WS action registrations."""

    def __init__(self) -> None:
        self.exact_actions: set[str] = set()
        self.prefix_actions: set[str] = set()
        self.handlers: dict[str, object] = {}
        self.replies: list[tuple[str, str]] = []
        self.enter_ttadk_calls: list[tuple] = []
        self._project_manager = MagicMock()
        self._project_manager.get_active_project.return_value = None

    def _register_action(self, handler, *, exact=None, prefix=None):
        if exact:
            self.exact_actions.add(exact)
            self.handlers[exact] = handler
        if prefix:
            self.prefix_actions.add(prefix)

    def _reply_text(self, message_id: str, text: str):
        self.replies.append((message_id, text))

    def _handle_card_enter_ttadk(self, *args):
        self.enter_ttadk_calls.append(args)

    def __getattr__(self, name):
        def _stub(*args, **kwargs):
            return None

        return _stub


# All WORKTREE_* constants from action_ids that should be in the registry.
_WORKTREE_ACTION_IDS = [
    action_ids.WORKTREE_FINISH_SELECTION,
    action_ids.WORKTREE_CONFIRM_START,
    action_ids.WORKTREE_MERGE,
    action_ids.WORKTREE_CLEANUP,
    action_ids.WORKTREE_RETRY_FAILED,
    action_ids.WORKTREE_RETRY_ALL,
    action_ids.WORKTREE_CANCEL,
    action_ids.SHOW_WORKTREE_MENU,
    # Common actions (inherited from build_common_action_registry)
    action_ids.MODE_FULL,
    action_ids.MODE_COMPACT,
    action_ids.ENGINE_STOP,
]


class TestBuildWorktreeActionRegistry:
    """Validate build_worktree_action_registry() coverage and correctness."""

    def test_all_worktree_ids_present(self):
        """Every expected worktree action_id has an entry in the registry."""
        registry = build_worktree_action_registry()
        for aid in _WORKTREE_ACTION_IDS:
            assert aid in registry, f"action_id {aid!r} missing from worktree registry"

    def test_no_extra_keys(self):
        """Registry contains only known worktree action_ids (no stale entries)."""
        registry = build_worktree_action_registry()
        expected = set(_WORKTREE_ACTION_IDS)
        extra = set(registry.keys()) - expected
        assert not extra, f"Unexpected keys in worktree registry: {extra}"

    @pytest.mark.parametrize("action_id", _WORKTREE_ACTION_IDS)
    def test_factory_returns_card_event(self, action_id: str):
        """Each factory in the registry returns a CardEvent when called with a dict payload."""
        registry = build_worktree_action_registry()
        factory = registry[action_id]
        event = factory({"test": True})
        assert isinstance(event, CardEvent), (
            f"factory for {action_id!r} returned {type(event).__name__}, expected CardEvent"
        )

    @pytest.mark.parametrize("action_id", _WORKTREE_ACTION_IDS)
    def test_factory_returns_card_event_with_empty_payload(self, action_id: str):
        """Factories handle empty payload without error."""
        registry = build_worktree_action_registry()
        factory = registry[action_id]
        event = factory({})
        assert isinstance(event, CardEvent)

    def test_cancel_factory_ignores_payload(self):
        """WORKTREE_CANCEL factory produces fixed payload regardless of input."""
        registry = build_worktree_action_registry()
        event = registry[action_ids.WORKTREE_CANCEL]({"arbitrary": "data"})
        assert event.payload == {"reason": "user_cancel"}


def test_feishu_action_registry_uses_canonical_core_action_ids():
    """WS action registry must stay aligned with card action-id constants."""
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)

    expected_exact = {
        action_ids.SHOW_STATUS,
        action_ids.SHOW_BOARD,
        action_ids.REFRESH_BOARD,
        action_ids.SWITCH_PROJECT,
        action_ids.SWITCH_BOARD_PAGE,
        action_ids.SHOW_DETAIL,
        action_ids.SWITCH_TO,
        action_ids.CONTINUE_DEV,
        action_ids.LIST_FILES,
        action_ids.NEW_PROJECT_PROMPT,
        action_ids.SHOW_HELP_MENU,
        action_ids.ENTER_DEEP_PROMPT,
        action_ids.SHOW_DEEP_STATUS,
        action_ids.RETRY_COMMAND,
        action_ids.CONTINUE_DEGRADED,
        action_ids.SHOW_ERROR_DETAILS,
        action_ids.RETRY_ORIGINAL,
        action_ids.HELP_CATEGORY,
        action_ids.SELECT_TTADK_TOOL,
        action_ids.TOGGLE_TTADK_YOLO,
        action_ids.SELECT_TTADK_MODEL,
        action_ids.REFRESH_TTADK_MODELS,
        action_ids.SELECT_TTADK_COMBINED,
        action_ids.SELECT_TTADK_COMBINED_TOOL,
        action_ids.SHOW_TTADK_MENU,
        action_ids.SHOW_ACP_MENU,
        action_ids.SELECT_ACP_TOOL,
        action_ids.SELECT_ACP_MODEL,
        action_ids.REFRESH_ACP_MODELS,
        action_ids.SLOCK_NEW_ROLE_SELECT_TOOL,
        action_ids.SLOCK_NEW_ROLE_SELECT_MODEL,
        action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_GROUP,
        action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_PROFILE,
        action_ids.SLOCK_NEW_ROLE_SELECT_MODEL_EFFORT,
        action_ids.SHOW_WORKTREE_MENU,
        action_ids.WORKTREE_FINISH_SELECTION,
        action_ids.WORKTREE_SELECT_TOOL,
        action_ids.WORKTREE_SELECT_MODEL,
        action_ids.WORKTREE_REMOVE_ITEM,
        action_ids.WORKTREE_CLEAR_ITEMS,
        action_ids.WORKTREE_CONFIRM_START,
        action_ids.WORKTREE_MERGE,
        action_ids.SHOW_WORKTREE_MERGE_ENTRY,
        action_ids.WORKTREE_CLEANUP,
        action_ids.WORKTREE_EXECUTE_ACTION,
        action_ids.WORKTREE_RETRY_FAILED,
        action_ids.WORKTREE_RETRY_ALL,
        action_ids.FORCE_RELEASE_REPO_LOCK,
        action_ids.CONFIRM_LOCK,
        action_ids.CANCEL_LOCK,
        action_ids.CONFIRM_FORCE_RELEASE,
        action_ids.CANCEL_FORCE_RELEASE,
        action_ids.APPROVE_ACTION,
        action_ids.REJECT_ACTION,
        action_ids.ENGINE_STOP,
    }

    assert expected_exact <= client.exact_actions
    assert {"deep_", "spec_", "slock_"} <= client.prefix_actions


@pytest.mark.parametrize(
    "action_type",
    (
        "employee_runtime_show_status",
        "employee_runtime_recycle_session",
        "employee_runtime_rebuild_workspace",
        "employee_runtime_lint_knowledge",
        "employee_runtime_retry_review",
    ),
)
def test_employee_runtime_action_dispatches_to_slock_handler(action_type: str) -> None:
    """Runtime status-card actions must be reachable from the global dispatcher."""
    from src.feishu.action_dispatcher import ActionDispatcher
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    client._slock_handler = MagicMock()
    dispatcher = ActionDispatcher()
    client._register_action = dispatcher.register
    init_action_registry(client)
    value = {"agent_id": "agt_atlas", "review_id": "review_1"}

    handled = dispatcher.dispatch(
        action_type,
        "message_1",
        "chat_1",
        None,
        value,
    )

    assert handled is True
    client._slock_handler.handle_card_action.assert_called_once_with(
        "message_1",
        "chat_1",
        action_type,
        value,
    )


def test_select_acp_model_default_option_passes_none_model():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    calls = []
    client._handle_select_acp_model = lambda *args: calls.append(args)
    init_action_registry(client)

    client.handlers[action_ids.SELECT_ACP_MODEL](
        "msg1",
        "chat1",
        None,
        {
            "tool_name": "codex",
            "model_name": "__ghostap_default_model__",
            "use_default_model": True,
        },
    )

    assert calls
    assert calls[0][2] == "codex"
    assert calls[0][3] is None


def test_degraded_continue_action_is_registered_separately_from_retry():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)

    assert action_ids.CONTINUE_DEGRADED in client.exact_actions
    assert action_ids.RETRY_COMMAND in client.exact_actions
    assert action_ids.SHOW_ERROR_DETAILS in client.exact_actions
    assert action_ids.RETRY_ORIGINAL in client.exact_actions
    assert action_ids.CONTINUE_DEGRADED != action_ids.RETRY_COMMAND


def test_continue_degraded_without_target_returns_unknown_mode_feedback():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)

    handler = client.handlers[action_ids.CONTINUE_DEGRADED]
    handler("m1", "c1", "p1", {})

    assert client.replies == [("m1", "当前暂未确定可继续模式，请重新发送原命令或查看诊断。")]


def test_show_error_details_action_replies_with_diagnostics():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)
    token = error_diagnostic_store.register(
        title="TTADK 暂不可用",
        summary="cli unavailable",
        details="stderr: boom at /home/alice/project/.env SECRET_TOKEN=abc123",
        chat_id="c1",
        origin_message_id="m1",
        request_id="req-1",
        trace_id="trace-1",
    )

    handler = client.handlers[action_ids.SHOW_ERROR_DETAILS]
    handler(
        "m1",
        "c1",
        "p1",
        {"diagnostic_token": token, "request_id": "req-1", "trace_id": "trace-1", "details": "payload must be ignored"},
    )

    assert len(client.replies) == 1
    reply = client.replies[0][1]
    assert "🔎 TTADK 暂不可用" in reply
    assert "cli unavailable" in reply
    assert "payload must be ignored" not in reply
    assert "/home/alice" not in reply
    assert "SECRET_TOKEN=abc123" not in reply
    assert "<path>" in reply
    assert "<redacted>" in reply


def test_show_error_details_action_rejects_mismatched_context():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)
    token = error_diagnostic_store.register(
        title="敏感诊断",
        summary="safe summary",
        details="secret detail",
        chat_id="c-allowed",
        origin_message_id="m-allowed",
        request_id="req-allowed",
    )

    handler = client.handlers[action_ids.SHOW_ERROR_DETAILS]
    handler("m-other", "c-other", "p1", {"diagnostic_token": token, "request_id": "req-allowed"})

    assert len(client.replies) == 1
    assert "无法查看该诊断详情" in client.replies[0][1]
    assert "secret detail" not in client.replies[0][1]


def test_show_error_details_action_uses_payload_origin_message_binding_for_card_click():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)
    token = error_diagnostic_store.register(
        title="原消息绑定诊断",
        summary="safe summary",
        details="origin-bound detail",
        chat_id="c1",
        origin_message_id="origin-msg",
        request_id="req-origin",
    )

    handler = client.handlers[action_ids.SHOW_ERROR_DETAILS]
    handler(
        "card-msg",
        "c1",
        "p1",
        {
            "diagnostic_token": token,
            "origin_message_id": "origin-msg",
            "request_id": "req-origin",
        },
    )

    assert len(client.replies) == 1
    assert "🔎 原消息绑定诊断" in client.replies[0][1]
    assert "origin-bound detail" in client.replies[0][1]


def test_retry_original_action_uses_use_case_without_private_ttadk_handler():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)

    handler = client.handlers[action_ids.RETRY_ORIGINAL]
    payload = {"original_mode": "ttadk_coco", "retry_mode": "ttadk_coco", "degraded_to": "Coco", "origin_message_id": "m0"}
    handler("m1", "c1", "p1", payload)

    assert client.enter_ttadk_calls == []
    assert client.replies == [
        (
            "m1",
            "已收到重试请求，但当前卡片无法安全自动恢复 ttadk_coco。请重新发送原命令、查看诊断，或在卡片存在可继续模式时使用该模式。",
        )
    ]


def test_retry_original_action_without_mode_returns_clear_feedback():
    from src.feishu.action_registry import init_action_registry

    client = _RegistryCaptureClient()
    init_action_registry(client)

    handler = client.handlers[action_ids.RETRY_ORIGINAL]
    handler("m1", "c1", "p1", {})

    assert client.replies == [("m1", "当前降级卡缺少可自动重试的原模式上下文，请重新发送原命令或查看诊断。")]


def test_chat_lock_exempt_actions_do_not_reference_stale_card_actions():
    """Chat-lock exemptions should not drift from canonical registered actions."""
    from src.chat_lock import ChatLockManager

    canonical_exempt = {
        action_ids.FORCE_RELEASE_REPO_LOCK,
        action_ids.CONFIRM_LOCK,
        action_ids.CANCEL_LOCK,
        action_ids.CONFIRM_FORCE_RELEASE,
        action_ids.CANCEL_FORCE_RELEASE,
        action_ids.HELP_CATEGORY,
        action_ids.RETRY_COMMAND,
    }

    assert ChatLockManager.CARD_EXEMPT_ACTIONS == canonical_exempt


def test_forwarding_map_handler_methods_are_validated_against_handler_classes():
    from src.feishu.router import validate_forwarding_map

    assert validate_forwarding_map() == []


def test_action_registry_dispatch_wiring_avoids_broad_exception_catches():
    """Action registry wiring is a core dispatch path; do not hide failures with broad catches."""
    path = Path(__file__).resolve().parents[1] / "src" / "feishu" / "action_registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    broad_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        typ = node.type
        is_broad = typ is None or (
            isinstance(typ, ast.Name) and typ.id in {"Exception", "BaseException"}
        )
        if is_broad:
            broad_lines.append(node.lineno)

    assert broad_lines == []
