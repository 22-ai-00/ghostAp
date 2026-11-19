from __future__ import annotations

import json
import time

from src.card.render.tools import _extract_json_field
from src.card.tool_display import (
    _MAX_OPAQUE_IDS,
    is_unhelpful_display_label,
    sanitize_full_tool_event_content,
    sanitize_full_tool_event_value,
    sanitize_tool_event_content,
    sanitize_tool_failure_detail,
    summarize_tool_call_content,
)
from src.workflow_engine.engine import _decode_result_payload


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


def test_full_tool_event_prevents_mapping_key_collisions() -> None:
    payload = {
        "sk-1234567890abcdefghijkl": "first-openai",
        "sk-abcdefghijklmnopqrstuv": "second-openai",
        "<redacted:api_key>": "literal-existing",
        "call_deadbeef1": "first-call",
        "call_deadbeef2": "second-call",
        "API_K\u200dEY": "joined-secret",
        "API_KEY": "existing-secret",
        "norm\u202ekey": "bidi-key",
        "normkey": "plain-key",
    }

    safe = sanitize_full_tool_event_value(payload)
    rendered = json.dumps(safe, ensure_ascii=False)

    assert safe["<redacted:api_key>"] == "<redacted>"
    assert safe["<redacted:api_key>#2"] == "first-openai"
    assert safe["<redacted:api_key>#3"] == "second-openai"
    assert safe["[redacted:call_id]"] == "first-call"
    assert safe["[redacted:call_id]#2"] == "second-call"
    assert safe["API_KEY"] == "<redacted>"
    assert safe["API_KEY#2"] == "<redacted>"
    assert safe["normkey"] == "plain-key"
    assert safe["normkey#2"] == "bidi-key"
    assert "sk-1234567890abcdefghijkl" not in rendered
    assert "sk-abcdefghijklmnopqrstuv" not in rendered
    assert "call_deadbeef1" not in rendered
    assert "call_deadbeef2" not in rendered
    assert "joined-secret" not in rendered
    assert "existing-secret" not in rendered


def test_full_tool_event_bounds_high_cardinality_opaque_ids_x_leaves() -> None:
    opaque_ids = [f"opaque-token-{index:04d}" for index in range(1000)]
    payload = {
        "items": [
            f"leaf-{index} opaque-token-0001 opaque-token-0255 opaque-token-0999"
            for index in range(1000)
        ]
    }

    started = time.monotonic()
    safe = sanitize_full_tool_event_value(payload, opaque_ids=opaque_ids)
    elapsed = time.monotonic() - started

    assert _MAX_OPAQUE_IDS == 256
    assert elapsed < 2.0
    assert len(safe["items"]) == 1000
    assert "opaque-token-0001" not in safe["items"][0]
    assert "opaque-token-0255" not in safe["items"][0]
    assert "opaque-token-0999" in safe["items"][0]


def test_full_tool_event_enforces_total_char_budget() -> None:
    under_limit = {f"k{index}": "a" * 50_000 for index in range(8)}
    safe_under = sanitize_full_tool_event_value(under_limit)
    assert isinstance(safe_under, dict)
    assert len(safe_under) == 8

    over_leaves = {f"k{index}": "a" * 60_000 for index in range(10)}
    assert sanitize_full_tool_event_value(over_leaves) == (
        "<redacted:structured_payload_limit>"
    )
    assert sanitize_full_tool_event_content(over_leaves) == (
        "<redacted:structured_payload_limit>"
    )

    over_keys = {f"key_{index}_{'k' * 60_000}": 1 for index in range(10)}
    assert sanitize_full_tool_event_value(over_keys) == (
        "<redacted:structured_payload_limit>"
    )


def test_deep_json_and_nested_structures_are_bounded_across_display_helpers() -> None:
    deep_array_1100 = "[" * 1100 + "0" + "]" * 1100
    deep_object_1100 = '{"a":' * 1100 + '"boom"' + "}" * 1100
    deep_array_10000 = "[" * 10000 + "0" + "]" * 10000
    deep_object_10000 = '{"a":' * 10000 + '"boom"' + "}" * 10000

    for payload in (
        deep_array_1100,
        deep_object_1100,
        deep_array_10000,
        deep_object_10000,
    ):
        assert summarize_tool_call_content(payload, fallback="fb")
        sanitized = sanitize_tool_event_content(payload, fallback="fb")
        assert sanitized in ("fb", payload)
        assert sanitize_tool_failure_detail(payload, fallback="fb")
        assert is_unhelpful_display_label(payload) is True
        assert _extract_json_field(payload, ("path", "file_path")) == ""
        assert isinstance(_extract_json_field(payload, ("a",)), str)
        _decode_result_payload(payload)

    deep_nested_dict: dict = {"error": "deep"}
    for _ in range(120):
        deep_nested_dict = {"wrapper": deep_nested_dict}
    assert sanitize_tool_failure_detail(deep_nested_dict, fallback="fb") == "fb"
