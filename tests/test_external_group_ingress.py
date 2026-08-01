from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.card.delivery.engine import CardDelivery
from src.card.events import CardEvent
from src.card.session import CardSession, SessionConfig
from src.card.session.static import StaticCardSession
from src.card.state.models import CardMetadata
from src.card.types import RenderedCard
from src.feishu.handlers.base import BaseHandler
from src.feishu.ws_card_action_handler import bind_managed_trust_revisions
from src.feishu.ws_client import FeishuWSClient
from src.slock_engine.activation_guard import ActivationGuard
from src.trust.models import ActorKind, EffectiveTrust, ManagedGroupOrigin, TrustZone
from src.trust.registry import ManagedGroupRegistry

OWNER_ID = "ou_owner"
GROUP_ID = "oc_managed"


def _registry(tmp_path) -> ManagedGroupRegistry:
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    registry.register(
        chat_id=GROUP_ID,
        owner_id=OWNER_ID,
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id="project-1",
        canonical_root_ref="/srv/project-1",
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        backend_binding_ids=("codex",),
    )
    return registry


def _revoke_registry(registry: ManagedGroupRegistry, state: str) -> None:
    if state == "revoking":
        registry.begin_revoke(
            GROUP_ID,
            requested_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
    else:
        registry.tombstone(GROUP_ID)


def _card_handler(registry) -> BaseHandler:
    handler = BaseHandler.__new__(BaseHandler)
    handler.ctx = SimpleNamespace(
        managed_group_registry=registry,
        message_linker=MagicMock(),
        settings=SimpleNamespace(default_reply_mode="thread"),
    )
    handler.ctx.message_linker.query.return_value = {"chat_id": GROUP_ID}
    handler.im_client = MagicMock()
    handler._resolve_origin = MagicMock(return_value="om_origin")
    handler._reply_audit_aliases = MagicMock(return_value=())
    handler.ensure_request_id = MagicMock(return_value="request-1")
    handler.format_ref_note = MagicMock(return_value="")
    handler._inject_ref_note = MagicMock(return_value="{}")
    handler._link_reply_response = MagicMock()
    return handler


def _handler_card_delivery(handler: BaseHandler) -> tuple[CardDelivery, MagicMock]:
    client = MagicMock()
    client.create_card.return_value = ("om_card", "om_card")
    delivery = CardDelivery(
        client,
        registry=MagicMock(),
        payload_transform=lambda chat_id, card: handler._bind_managed_card_revisions(
            card,
            chat_id=chat_id,
        ),
        trust_revision_provider=handler._managed_card_trust_revisions,
    )
    return delivery, client


def _message(*, chat_id: str, sender_id: str) -> MagicMock:
    data = MagicMock()
    data.header.tenant_key = "tenant-1"
    data.event.message.message_id = "om_external"
    data.event.message.chat_id = chat_id
    data.event.message.chat_type = "group"
    data.event.message.create_time = "9999999999999"
    data.event.message.message_type = "image"
    data.event.message.content = '{"image_key":"img_external"}'
    data.event.message.parent_id = None
    data.event.message.root_id = None
    data.event.sender.sender_id.open_id = sender_id
    data.event.sender.sender_id.union_id = "on_external"
    return data


def _base_client(registry: ManagedGroupRegistry) -> FeishuWSClient:
    client = FeishuWSClient.__new__(FeishuWSClient)
    client.settings = SimpleNamespace(
        admin_user_ids=frozenset({OWNER_ID}),
        allowed_user_ids=frozenset({OWNER_ID, "ou_unknown"}),
        allowed_chat_ids=frozenset({GROUP_ID, "oc_external"}),
        ingress_access_mode="legacy_allow_all",
        admin_bootstrap_scope="p2p_only",
        thread_programming_enabled=False,
    )
    client._managed_group_registry = registry
    client._managed_group_owner_id = OWNER_ID
    client._employee_department_runtime = MagicMock()
    client._employee_department_runtime.trusted_employee_bot_open_ids.return_value = ()
    client._scheduler = MagicMock()
    client._scheduler.submit.return_value = SimpleNamespace(run_id="run-1")
    client._message_linker = MagicMock()
    client._message_linker.resolve_origin.return_value = None
    client._message_mapper = MagicMock()
    client._message_mapper.get_project_id.return_value = None
    client._project_manager = MagicMock()
    client._project_manager.get_active_project.return_value = None
    client._thread_manager = MagicMock()
    client._ensure_request_id = MagicMock(return_value="req-external")
    client._extract_text_from_message = MagicMock(return_value="do dangerous work")
    client._is_exit_command = MagicMock(return_value=False)
    client._is_spec_command = MagicMock(return_value=False)
    client._build_control_queue_key = MagicMock(return_value=None)
    client._get_image_handler = MagicMock()
    client._get_image_handler.return_value.parse_message.return_value = SimpleNamespace(
        text="do dangerous work",
        image_keys=["img_external"],
    )
    client._validate_message = MagicMock(return_value=True)
    client._chat_lock_gate = MagicMock()
    client._handle_image_content = MagicMock()
    client._resolve_message_context = MagicMock(return_value=(None, None))
    client._dispatch_message_logic = MagicMock()
    client._dispatch_empty_text = MagicMock()
    client._get_api_client = MagicMock()
    client._pending_image_lock = nullcontext()
    client._pending_image_keys = {}
    client._pending_image_only = set()
    return client


@pytest.mark.parametrize(
    ("chat_id", "sender_id"),
    [
        ("oc_external", OWNER_ID),
        (GROUP_ID, "ou_unknown"),
    ],
)
def test_unknown_member_or_external_group_has_zero_business_side_effects(
    tmp_path,
    chat_id: str,
    sender_id: str,
) -> None:
    client = _base_client(_registry(tmp_path))
    data = _message(chat_id=chat_id, sender_id=sender_id)

    client._handle_message(data)
    client._process_message_async(data)

    client._extract_text_from_message.assert_not_called()
    client._get_image_handler.assert_not_called()
    client._project_manager.get_active_project.assert_not_called()
    client._employee_department_runtime.record_group_event.assert_not_called()
    client._handle_image_content.assert_not_called()
    client._dispatch_message_logic.assert_not_called()
    client._scheduler.submit.assert_not_called()
    client._message_linker.register_origin.assert_not_called()


def _card(*, chat_id: str, operator_id: str, revisions: tuple[int, int]) -> MagicMock:
    data = MagicMock()
    data.header.event_id = "evt-card"
    data.header.event_type = "card.action.trigger"
    data.header.tenant_key = "tenant-1"
    data.event.context.open_message_id = "om_card"
    data.event.context.open_chat_id = chat_id
    data.event.operator.open_id = operator_id
    data.event.operator.user_id = None
    data.event.operator.union_id = "on_operator"
    data.event.action.tag = "button"
    data.event.action.name = "execute"
    data.event.action.behaviors = []
    data.event.action.value = {
        "action": "workflow_confirm_start",
        "project_id": "project-1",
        "group_revision": revisions[0],
        "grant_revision": revisions[1],
    }
    return data


def _card_client(registry: ManagedGroupRegistry) -> FeishuWSClient:
    client = _base_client(registry)
    client._card_event_cache = MagicMock()
    client._card_event_cache.is_duplicate.return_value = False
    client._card_action_dedup_cache = MagicMock()
    client._card_action_dedup_cache.is_duplicate.return_value = False
    client._system_cmd_gate_lock = nullcontext()
    client._system_cmd_inflight_by_chat = {}
    client._resolve_card_is_p2p = MagicMock(return_value=False)
    client._is_system_card_action = MagicMock(return_value=True)
    return client


def test_stale_card_revision_cannot_dispatch_effect(tmp_path) -> None:
    registry = _registry(tmp_path)
    current_group = registry.active_record(GROUP_ID)
    current_grant = registry.grant_for_chat(GROUP_ID)
    assert current_group is not None and current_grant is not None
    client = _card_client(registry)
    client._refresh_managed_card_revisions = MagicMock(return_value=True)
    data = _card(
        chat_id=GROUP_ID,
        operator_id=OWNER_ID,
        revisions=(current_group.revision + 1, current_grant.revision),
    )

    client._handle_card_action(data)

    client._card_event_cache.is_duplicate.assert_not_called()
    client._card_action_dedup_cache.is_duplicate.assert_not_called()
    client._project_manager.get_active_project.assert_not_called()
    client._message_linker.resolve_origin.assert_not_called()
    client._scheduler.submit.assert_not_called()
    client._refresh_managed_card_revisions.assert_called_once()


def test_card_rotation_after_enqueue_is_fenced_before_action_dispatch(tmp_path) -> None:
    registry = _registry(tmp_path)
    client = _card_client(registry)
    client._action_dispatcher = MagicMock()
    client._refresh_managed_card_revisions = MagicMock(return_value=True)
    intake_trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    assert intake_trust is not None
    registry.rotate_receiving_bot(
        chat_id=GROUP_ID,
        expected_bot_ref="cli_main_bot",
        new_bot_ref="cli_rotated_bot",
    )
    current = registry.active_record(GROUP_ID)
    grant = registry.grant_for_chat(GROUP_ID)
    assert current is not None and grant is not None
    data = _card(
        chat_id=GROUP_ID,
        operator_id=OWNER_ID,
        revisions=(current.revision, grant.revision),
    )

    client._process_card_action_async(data, effective_trust=intake_trust)

    client._action_dispatcher.dispatch.assert_not_called()
    client._chat_lock_gate.check_card_action.assert_not_called()
    client._refresh_managed_card_revisions.assert_called_once()


def test_stale_card_refresh_reads_existing_card_and_only_updates_revisions(tmp_path) -> None:
    registry = _registry(tmp_path)
    client = _card_client(registry)
    trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    card = {
        "schema": "2.0",
        "body": {
            "elements": [
                {"tag": "button", "value": {"action": "workflow_confirm_start"}}
            ]
        },
    }
    item = SimpleNamespace(
        message_id="om_card",
        chat_id=GROUP_ID,
        body=SimpleNamespace(content=__import__("json").dumps(card)),
    )
    response = MagicMock()
    response.success.return_value = True
    response.data = SimpleNamespace(items=[item])
    client._get_api_client.return_value.im.v1.message.get.return_value = response
    client._system_handler = MagicMock()
    client._system_handler.update_card.return_value = True

    assert client._refresh_managed_card_revisions("om_card", GROUP_ID, trust) is True

    refreshed = client._system_handler.update_card.call_args.args[1]
    value = refreshed["body"]["elements"][0]["value"]
    assert value["action"] == "workflow_confirm_start"
    assert value["group_revision"] == trust.group_revision
    assert value["grant_revision"] == trust.grant_revision
    client._scheduler.submit.assert_not_called()


def test_external_card_callback_is_silent_before_mutation(tmp_path) -> None:
    registry = _registry(tmp_path)
    client = _card_client(registry)
    data = _card(
        chat_id="oc_external",
        operator_id="ou_unknown",
        revisions=(1, 1),
    )

    result = client._handle_card_action(data)

    assert result is None
    client._card_event_cache.is_duplicate.assert_not_called()
    client._card_action_dedup_cache.is_duplicate.assert_not_called()
    client._project_manager.get_active_project.assert_not_called()
    client._message_linker.resolve_origin.assert_not_called()
    client._scheduler.submit.assert_not_called()


def test_managed_card_without_issued_revision_is_stale(tmp_path) -> None:
    registry = _registry(tmp_path)
    client = _card_client(registry)
    data = _card(
        chat_id=GROUP_ID,
        operator_id=OWNER_ID,
        revisions=(1, 1),
    )
    del data.event.action.value["group_revision"]
    del data.event.action.value["grant_revision"]

    client._handle_card_action(data)

    client._card_event_cache.is_duplicate.assert_not_called()
    client._scheduler.submit.assert_not_called()


def test_production_card_values_receive_managed_revision_snapshot() -> None:
    card = {
        "schema": "2.0",
        "body": {
            "elements": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "Run"},
                    "value": {"action": "workflow_confirm_start"},
                    "behaviors": [
                        {
                            "type": "callback",
                            "value": {"action": "workflow_confirm_start"},
                        }
                    ],
                }
            ]
        },
    }

    bound = bind_managed_trust_revisions(
        card,
        group_revision=7,
        grant_revision=11,
    )

    button = bound["body"]["elements"][0]
    assert button["value"]["group_revision"] == 7
    assert button["value"]["grant_revision"] == 11
    assert button["behaviors"][0]["value"]["group_revision"] == 7
    assert "group_revision" not in card["body"]["elements"][0]["value"]


def test_card_delivery_stamps_rendered_payload_before_transport() -> None:
    client = MagicMock()
    client.create_card.return_value = ("om_card", "om_card")
    delivery = CardDelivery(
        client,
        registry=MagicMock(),
        payload_transform=lambda _chat_id, card: bind_managed_trust_revisions(
            card,
            group_revision=7,
            grant_revision=11,
        ),
    )
    rendered = RenderedCard(
        _card_json={
            "schema": "2.0",
            "body": {
                "elements": [
                    {
                        "tag": "button",
                        "value": {"action": "workflow_confirm_start"},
                    }
                ]
            },
        },
        structure_signature="stable",
    )

    delivery.deliver("session-1", GROUP_ID, [rendered])

    payload = client.create_card.call_args.args[1]
    assert payload["body"]["elements"][0]["value"] == {
        "action": "workflow_confirm_start",
        "group_revision": 7,
        "grant_revision": 11,
    }


def test_long_card_run_cannot_be_reblessed_after_rotation() -> None:
    client = MagicMock()
    client.create_card.return_value = ("om_card", "om_card")
    revision = [7, 11]
    delivery = CardDelivery(
        client,
        registry=MagicMock(),
        payload_transform=lambda _chat_id, card: bind_managed_trust_revisions(
            card,
            group_revision=revision[0],
            grant_revision=revision[1],
        ),
        trust_revision_provider=lambda _chat_id: tuple(revision),
    )
    rendered = RenderedCard(
        _card_json={
            "schema": "2.0",
            "body": {"elements": [{"tag": "button", "value": {"action": "run"}}]},
        },
        structure_signature="stable",
    )

    delivery.deliver("session-rotation", GROUP_ID, [rendered])
    revision[0] += 1
    outcomes = delivery.deliver("session-rotation", GROUP_ID, [rendered])

    assert client.create_card.call_count == 1
    client.update_card.assert_not_called()
    assert outcomes and outcomes[0].kind == "rejected"


@pytest.mark.parametrize("next_snapshot", [None, RuntimeError("registry unavailable")])
def test_card_session_start_snapshot_fails_closed_before_first_send(next_snapshot) -> None:
    client = MagicMock()
    client.create_card.return_value = ("om_card", "om_card")
    current = [(7, 11)]

    def snapshot(_chat_id):
        value = current[0]
        if isinstance(value, Exception):
            raise value
        return value

    delivery = CardDelivery(
        client,
        registry=MagicMock(),
        payload_transform=lambda _chat_id, card: bind_managed_trust_revisions(
            card,
            group_revision=7,
            grant_revision=11,
        ),
        trust_revision_provider=snapshot,
    )
    session = StaticCardSession(delivery, GROUP_ID, session_id="session-start")
    current[0] = next_snapshot

    assert session.send({"schema": "2.0", "body": {"elements": []}}) is None
    client.create_card.assert_not_called()


@pytest.mark.parametrize("registry_state", ["revoking", "tombstoned"])
@pytest.mark.parametrize("session_kind", ["continuation", "static"])
def test_new_card_sessions_fail_closed_after_managed_group_revoke(
    tmp_path,
    registry_state: str,
    session_kind: str,
) -> None:
    registry = _registry(tmp_path)
    _revoke_registry(registry, registry_state)
    handler = _card_handler(registry)
    delivery, client = _handler_card_delivery(handler)

    if session_kind == "continuation":
        session = CardSession(
            GROUP_ID,
            SessionConfig(
                metadata=CardMetadata(mode_name="Test"),
                sync_delivery=True,
            ),
            delivery,
            session_id=f"{registry_state}-continuation",
        )
        session.dispatch(CardEvent.started())
    else:
        session = StaticCardSession(
            delivery,
            GROUP_ID,
            session_id=f"{registry_state}-static",
        )
        assert session.send({"schema": "2.0", "body": {"elements": []}}) is None

    client.create_card.assert_not_called()
    client.update_card.assert_not_called()
    session.close()


@pytest.mark.parametrize("registry_state", ["revoking", "tombstoned"])
def test_direct_base_handler_cards_fail_closed_after_managed_group_revoke(
    tmp_path,
    registry_state: str,
) -> None:
    registry = _registry(tmp_path)
    _revoke_registry(registry, registry_state)
    handler = _card_handler(registry)
    card = {"schema": "2.0", "body": {"elements": []}}

    assert handler.reply_card("om_origin", card) is None
    assert handler.update_card("om_card", card) is False
    assert handler.send_card_to_chat(GROUP_ID, card) is None

    handler.im_client.reply_message.assert_not_called()
    handler.im_client.patch_message.assert_not_called()
    handler.im_client.send_message.assert_not_called()


def test_direct_base_handler_rejects_issued_stamp_without_current_snapshot(
    tmp_path,
) -> None:
    registry = _registry(tmp_path)
    group, grant = registry.trust_snapshot(GROUP_ID)
    assert group is not None and grant is not None
    stamped = bind_managed_trust_revisions(
        {
            "schema": "2.0",
            "body": {
                "elements": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Run"},
                        "value": {"action": "run"},
                    }
                ]
            },
        },
        group_revision=group.revision,
        grant_revision=grant.revision,
    )
    handler = _card_handler(registry)

    assert handler.send_card_to_chat("oc_never_managed", stamped) is None
    handler.im_client.send_message.assert_not_called()


def test_direct_base_handler_fails_closed_on_snapshot_read_error() -> None:
    registry = MagicMock()
    registry.trust_snapshot.side_effect = OSError("registry unavailable")
    handler = _card_handler(registry)
    card = {"schema": "2.0", "body": {"elements": []}}

    assert handler.reply_card("om_origin", card) is None
    assert handler.update_card("om_card", card) is False
    assert handler.send_card_to_chat(GROUP_ID, card) is None

    handler.im_client.reply_message.assert_not_called()
    handler.im_client.patch_message.assert_not_called()
    handler.im_client.send_message.assert_not_called()


def test_activation_guard_denies_external_trust_before_rate_limit() -> None:
    guard = ActivationGuard()
    settings = SimpleNamespace(
        admin_user_ids=frozenset({OWNER_ID}),
        slock_auto_activate_whitelist_user_ids="",
        slock_auto_activate_default_policy="allow_all",
        slock_passive_mode=True,
        slock_auto_activate_rate_limit_per_user=3,
        slock_auto_activate_rate_limit_global=10,
    )
    external = EffectiveTrust(
        zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
        actor=ActorKind.UNKNOWN,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )

    allowed, reason = guard.can_auto_activate(
        OWNER_ID,
        "oc_external",
        settings,
        effective_trust=external,
    )

    assert allowed is False
    assert reason == "external_or_unknown"
    assert guard._user_timestamps == {}
