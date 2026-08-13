from __future__ import annotations

import json

from src.card.tool_display import (
    sanitize_full_tool_event_content,
    sanitize_full_tool_event_value,
)


def test_full_tool_event_structured_payload_is_sanitized_before_json_dump() -> None:
    payload = {
        "message": "API_KEY=secret2",
        "tail": "TAIL",
        "api_key": "api-secret",
        "access_token": "access-secret",
        "password": "password-secret",
    }

    rendered = sanitize_full_tool_event_content(payload)
    decoded = json.loads(rendered)

    assert decoded == {
        "message": "API_KEY=<redacted>",
        "tail": "TAIL",
        "api_key": "<redacted>",
        "access_token": "<redacted>",
        "password": "<redacted>",
    }
    assert rendered.endswith('"}')
    assert "secret2" not in rendered


def test_full_tool_event_sensitive_key_matching_uses_semantic_boundaries() -> None:
    payload = {
        "apiKey": "api-secret",
        "access-token": "access-secret",
        "clientsecret": "client-secret",
        "databasepassword": "database-secret",
        "awssecretaccesskey": "aws-secret",
        "secretkey": "generic-secret",
        "api_keys": ["api-secret-1"],
        "access_tokens": ["access-secret-1"],
        "passwords": ["password-secret-1"],
        "secrets": ["generic-secret-1"],
        "cookies": ["cookie-secret-1"],
        "primaryapikey": "primary-api-secret",
        "backupaccesstoken": "backup-access-secret",
        "oauthrefreshtoken": "oauth-refresh-secret",
        "mysessiontoken": "session-secret",
        "myprivatekey": "private-secret",
        "key_files": ["public.pem"],
        "key_findings": ["visible finding"],
        "keyboard_layout": "us",
        "monkey_patch": True,
        "hockey_score": "3-2",
        "token_usage": 42,
    }

    safe = sanitize_full_tool_event_value(payload)

    assert safe == {
        "apiKey": "<redacted>",
        "access-token": "<redacted>",
        "clientsecret": "<redacted>",
        "databasepassword": "<redacted>",
        "awssecretaccesskey": "<redacted>",
        "secretkey": "<redacted>",
        "api_keys": "<redacted>",
        "access_tokens": "<redacted>",
        "passwords": "<redacted>",
        "secrets": "<redacted>",
        "cookies": "<redacted>",
        "primaryapikey": "<redacted>",
        "backupaccesstoken": "<redacted>",
        "oauthrefreshtoken": "<redacted>",
        "mysessiontoken": "<redacted>",
        "myprivatekey": "<redacted>",
        "key_files": ["public.pem"],
        "key_findings": ["visible finding"],
        "keyboard_layout": "us",
        "monkey_patch": True,
        "hockey_score": "3-2",
        "token_usage": 42,
    }


def test_full_tool_event_preserves_internal_identifier_and_path_redaction() -> None:
    payload = {
        "thread_id": "thread-private",
        "subagent_path": "/private/agent/path",
        "agents_states": {
            "source-private": {
                "path": "/private/state/path",
                "status": "running",
            }
        },
        "echo": (
            "thread-private /private/agent/path /private/state/path "
            "source-private call_deadbeef"
        ),
    }

    safe = sanitize_full_tool_event_value(payload)
    rendered = json.dumps(safe, ensure_ascii=False)

    assert safe["thread_id"] == "<redacted:agent_id>"
    assert safe["subagent_path"] == "<redacted:agent_path>"
    assert list(safe["agents_states"]) == ["<redacted:agent_id>"]
    assert safe["agents_states"]["<redacted:agent_id>"]["path"] == (
        "<redacted:agent_path>"
    )
    for private_value in (
        "thread-private",
        "/private/agent/path",
        "/private/state/path",
        "source-private",
        "call_deadbeef",
    ):
        assert private_value not in rendered


def test_full_tool_event_deep_json_returns_safe_fallback() -> None:
    deeply_nested_json = "[" * 1100 + "0" + "]" * 1100

    assert sanitize_full_tool_event_value(deeply_nested_json) == (
        "<redacted:structured_payload_limit>"
    )
    assert sanitize_full_tool_event_content(deeply_nested_json) == (
        "<redacted:structured_payload_limit>"
    )


def test_full_tool_event_large_structure_returns_safe_fallback() -> None:
    oversized_structure = list(range(10_001))

    assert sanitize_full_tool_event_value(oversized_structure) == (
        "<redacted:structured_payload_limit>"
    )
    assert sanitize_full_tool_event_content(oversized_structure) == (
        "<redacted:structured_payload_limit>"
    )


def test_full_tool_event_preserves_emoji_joiners_but_removes_bidi_controls() -> None:
    safe = sanitize_full_tool_event_value(
        {
            "label": "家庭 👨‍👩‍👧‍👦",
            "unsafe": "前\u202e后",
            "message": "家庭 👨‍👩‍👧‍👦 pass‍word=visible-secret",
            "API_K‍EY": "hidden-secret",
        }
    )

    assert safe["label"] == "家庭 👨‍👩‍👧‍👦"
    assert safe["unsafe"] == "前后"
    assert safe["message"] == "家庭 👨‍👩‍👧‍👦 password=<redacted>"
    assert safe["API_KEY"] == "<redacted>"
    assert "visible-secret" not in json.dumps(safe, ensure_ascii=False)
    assert "hidden-secret" not in json.dumps(safe, ensure_ascii=False)


def test_full_tool_event_recursively_sanitizes_nested_json_strings() -> None:
    safe = sanitize_full_tool_event_value(
        {"nested": r'{"api_\u006bey":"nested-secret","keep":"ok"}'}
    )

    assert json.loads(safe["nested"]) == {
        "api_key": "<redacted>",
        "keep": "ok",
    }
    assert "nested-secret" not in json.dumps(safe, ensure_ascii=False)


def test_full_tool_event_sanitizes_credentials_embedded_in_mapping_keys() -> None:
    payload = {
        "sk-1234567890abcdefghijkl": "openai-key-name",
        "ghp_1234567890abcdefghijklmnop": "github-key-name",
        "Bearer token-value": "bearer-key-name",
        "ordinary_field": "visible",
    }

    safe = sanitize_full_tool_event_value(payload)
    rendered = json.dumps(safe, ensure_ascii=False)

    assert safe["<redacted:api_key>"] == "openai-key-name"
    assert safe["<redacted:github_token>"] == "github-key-name"
    assert safe["Bearer <redacted>"] == "bearer-key-name"
    assert safe["ordinary_field"] == "visible"
    assert "sk-1234567890abcdefghijkl" not in rendered
    assert "ghp_1234567890abcdefghijklmnop" not in rendered
    assert "token-value" not in rendered


def test_full_tool_event_replaces_lone_surrogates_and_preserves_valid_pairs() -> None:
    safe = sanitize_full_tool_event_value(
        {
            "bad\ud800field": "left\udfff-right",
            "encoded_pair": "\ud83d\ude00",
        }
    )
    rendered = sanitize_full_tool_event_content(safe)

    assert safe == {
        "bad�field": "left�-right",
        "encoded_pair": "😀",
    }
    assert "\ud800" not in rendered
    assert "\udfff" not in rendered
    rendered.encode("utf-8")
