import json
from unittest.mock import MagicMock, patch

from src.feishu.ws_client import FeishuWSClient
from src.utils.rate_limit import RateLimitExceededException


@patch("src.feishu.ws_client.ProjectManager")
@patch("src.feishu.ws_client.TaskScheduler")
def test_spec_backpressure(
    mock_scheduler_cls,
    _project_manager_cls,
    monkeypatch,
    tmp_path,
):
    from src.access_control import IngressAccessPolicy, IngressAccessPolicyProvider
    from src.config import IngressAccessMode
    from src.feishu import ws_client as ws

    mock_scheduler = mock_scheduler_cls.return_value
    settings = ws.get_settings().model_copy(update={"restart_gate_dir": ""})
    monkeypatch.setattr(ws, "get_settings", lambda: settings)
    monkeypatch.setattr(ws, "_CHECKOUT_ROOT", tmp_path)

    # Simulate backpressure by raising the exception on submit for spec tasks
    def mock_submit(spec, fn):
        if spec.task_type == "spec_command":
            raise RateLimitExceededException("Rate limit exceeded")
        return MagicMock()

    mock_scheduler.submit.side_effect = mock_submit

    client = FeishuWSClient(message_callback=lambda x: None)
    reply_text = MagicMock()
    client._handler_ctx.handlers["coco"].reply_text = reply_text
    # This unit test exercises scheduler backpressure after trust admission.
    # Registry-first ingress behavior has its own managed/external group suite.
    client._managed_group_registry = None
    client._ingress_access_policy_provider = IngressAccessPolicyProvider(
        IngressAccessPolicy(
            admin_ids=frozenset(),
            allowed_user_ids=frozenset({"ou_user"}),
            allowed_chat_ids=frozenset({"oc_chat"}),
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope="p2p_only",
        )
    )

    data = MagicMock()
    data.event.message.message_id = "om_1"
    data.event.message.chat_id = "oc_chat"
    data.event.message.chat_type = "group"
    data.event.message.message_type = "text"
    data.event.message.content = json.dumps({"text": "/spec do something"})
    data.event.sender.sender_id.open_id = "ou_user"

    client._handle_message(data)

    reply_text.assert_called_once()
    args = reply_text.call_args[0]
    assert "系统繁忙" in args[1]
