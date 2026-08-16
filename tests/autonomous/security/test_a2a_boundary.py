from __future__ import annotations

import json
import traceback
from types import SimpleNamespace

import pytest
from a2a.types import Message, Part, Role

from src.autonomous.a2a.card import (
    AgentCardValidationError,
    PilotAgentRegistration,
    canonical_card_digest,
    load_trusted_agent_card,
)
from src.autonomous.a2a.codec import A2ACodecError, normalize_a2a_observation

_CARD_URL = "https://cards.example.test/.well-known/agent-card.json"
_ENDPOINT_URL = "https://agent.example.test/a2a"


def _card(endpoint: str = _ENDPOINT_URL) -> bytes:
    return json.dumps(
        {
            "name": "reviewer",
            "description": "Untrusted public description",
            "supportedInterfaces": [
                {
                    "url": endpoint,
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": "1.0",
                }
            ],
            "version": "1",
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [
                {
                    "id": "review",
                    "name": "Review",
                    "description": "Review an anchored change",
                    "tags": ["review"],
                }
            ],
        }
    ).encode()


def _registration(raw: bytes, **overrides: str) -> PilotAgentRegistration:
    values = {
        "tenant_key": "tenant-a",
        "agent_id": "agt_agent-a",
        "card_url": _CARD_URL,
        "endpoint_url": _ENDPOINT_URL,
        "expected_card_digest": canonical_card_digest(raw),
        "credential_ref": "cred_a2a_agent-a",
    }
    values.update(overrides)
    return PilotAgentRegistration(**values)


@pytest.mark.parametrize(
    "url",
    [
        "http://agent.example.test/a2a",
        "https://user:password@agent.example.test/a2a",
        "https://agent.example.test/a2a#alternate",
        "https://localhost/a2a",
        "https://api.localhost/a2a",
        "https://127.0.0.1/a2a",
        "https://127.1/a2a",
        "https://0x7f000001/a2a",
        "https://10.1.2.3/a2a",
        "https://169.254.169.254/latest/meta-data",
        "https://192.168.1.2/a2a",
        "https://[::1]/a2a",
        "https://[fe80::1]/a2a",
        "https://[2001:db8::1]/a2a",
        "https://agent.example.test./a2a",
        "https://agent_example.test/a2a",
        "https://agent.example.test\\@127.0.0.1/a2a",
        "https://agent.example.test/\nheader",
    ],
)
def test_registration_rejects_non_public_or_ambiguous_urls(url: str) -> None:
    raw = _card()
    with pytest.raises(AgentCardValidationError, match="unsafe-url"):
        _registration(raw, endpoint_url=url)


def test_card_cannot_replace_authoritative_endpoint_even_with_matching_digest() -> None:
    drifted = _card("https://attacker.example.test/a2a")
    with pytest.raises(AgentCardValidationError, match="registered-interface-missing"):
        load_trusted_agent_card(_registration(drifted), drifted)


def test_public_card_has_only_credential_reference_not_credential_material() -> None:
    raw = _card()
    trusted = load_trusted_agent_card(_registration(raw), raw)
    assert trusted.registration.credential_ref == "cred_a2a_agent-a"
    assert b"cred_a2a_agent-a" not in trusted.sdk_card.SerializeToString()
    assert not any("credential" in field.name for field in trusted.sdk_card.DESCRIPTOR.fields)


def test_card_interface_tenant_cannot_override_registration() -> None:
    payload = json.loads(_card())
    payload["supportedInterfaces"][0]["tenant"] = "remote-tenant"
    raw = json.dumps(payload).encode()

    with pytest.raises(AgentCardValidationError, match="registered-interface-missing"):
        load_trusted_agent_card(_registration(raw), raw)


def test_remote_url_detail_is_not_reflected_in_codec_exception() -> None:
    secret_url = "https://secret.example.test/object?bearer=do-not-log"
    message = Message(
        message_id="message-1",
        context_id="ctx-1",
        task_id="task-1",
        role=Role.ROLE_AGENT,
        parts=[Part(url=secret_url)],
    )
    with pytest.raises(A2ACodecError) as exc_info:
        normalize_a2a_observation(
            message,
            SimpleNamespace(context_id="ctx-1", task_id="task-1"),
        )
    assert secret_url not in str(exc_info.value)
    assert "do-not-log" not in str(exc_info.value)


def test_remote_identifier_detail_is_not_reflected_in_codec_exception() -> None:
    remote_secret_id = "task-bearer-do-not-log"
    message = Message(
        message_id="message-1",
        context_id="ctx-1",
        task_id=remote_secret_id,
        role=Role.ROLE_AGENT,
        parts=[Part(text="result")],
    )
    with pytest.raises(A2ACodecError) as exc_info:
        normalize_a2a_observation(
            message,
            SimpleNamespace(context_id="ctx-1", task_id="task-1"),
        )
    assert remote_secret_id not in str(exc_info.value)


def test_card_url_parser_cause_does_not_leak_untrusted_port_text() -> None:
    raw = _card()
    secret_port = "do-not-log-secret-port"

    with pytest.raises(AgentCardValidationError) as exc_info:
        _registration(
            raw,
            endpoint_url=f"https://agent.example.test:{secret_port}/a2a",
        )

    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert secret_port not in rendered
