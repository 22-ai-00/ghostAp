"""Tests for lazy loading of ``lark_oapi`` packages and ``Client``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from src.utils.lark_lazy import _model_module_candidates


def test_model_module_candidates_handles_models_builders_and_events():
    assert "create_message_request" in _model_module_candidates("CreateMessageRequest")
    assert "create_message_request" in _model_module_candidates("CreateMessageRequestBuilder")
    assert "app_slash_command_i18n" in _model_module_candidates("AppSlashCommandI18n")
    assert "p2_contact_employee_type_enum_deactivated_v3" in _model_module_candidates(
        "P2ContactEmployeeTypeEnumDeactivatedV3Data"
    )


def test_install_lazy_lark_oapi_isolated_subprocess():
    """Fresh process using install_lazy_lark_oapi loads only accessed endpoints."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
from unittest.mock import MagicMock, patch

from src.utils.lark_lazy import install_lazy_lark_oapi

assert install_lazy_lark_oapi() is True
# Idempotent re-call
assert install_lazy_lark_oapi() is True

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetChatRequest,
    GetMessageRequest,
)
from lark_oapi.api.im.v1.model.reply_message_request import ReplyMessageRequest
from lark_oapi.api.cardkit.v1 import Card, UpdateCardRequest, CreateCardRequest
from lark_oapi.api.application.v7 import AppSlashCommand, AppSlashCommandI18n, AppSlashCommandI18nText
from lark_oapi.api.contact.v3 import GetUserRequest

# Lazy root exports core & builder symbols
assert lark.LogLevel.INFO is not None
assert lark.JSON is not None
assert lark.EventDispatcherHandler is not None

# Unbuilt client matches original: service attrs are None, private raises AttributeError
unbuilt = lark.Client()
assert unbuilt.im is None
assert unbuilt.config is None

# Built client resolves only accessed service/version/resource lazily
client = (
    lark.Client.builder()
    .app_id("cli_test")
    .app_secret("secret_test")
    .timeout(15)
    .log_level(lark.LogLevel.INFO)
    .build()
)
assert isinstance(client, lark.Client)
assert client.config is not None
assert client.config.app_id == "cli_test"
assert hasattr(client.im.v1.message, "create")
assert hasattr(client.im.v1.chat, "get")
assert hasattr(client.im.v1.chat_managers, "add_managers")
assert hasattr(client.im.v1.chat_members, "is_in_chat")
assert hasattr(client.cardkit.v1.card, "create")
assert hasattr(client.cardkit.v1.card_element, "content")
assert hasattr(client.contact.v3.user, "get")

# Unknown attributes raise AttributeError (never swallow typos or mock dunders)
for target, attr in [
    (lark, "DefinitelyMissingAttr"),
    (client, "_nonexistent_private"),
    (client, "nonexistent_service"),
    (client.im, "v99"),
    (client.im.v1, "nonexistent_resource"),
]:
    try:
        getattr(target, attr)
    except AttributeError:
        pass
    else:
        raise AssertionError(f"Expected AttributeError for {target!r}.{attr}")

# Patching client resources works like standard Client
fake_msg = MagicMock()
with patch.object(client.im.v1, "message", fake_msg):
    assert client.im.v1.message is fake_msg

# Unused API services without event processors remain unloaded even after EventDispatcherHandler access
assert "lark_oapi.api.baike" not in sys.modules
assert "lark_oapi.api.docs" not in sys.modules
assert "lark_oapi.api.sheets" not in sys.modules
assert "lark_oapi.api.im.v1.model.list_pin_request" not in sys.modules
assert len(sys.modules) < 2000

print(json.dumps({"modules": len(sys.modules)}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"


def test_install_lazy_lark_oapi_returns_false_when_real_sdk_loaded_early():
    """If real lark_oapi is already in sys.modules, install_lazy_lark_oapi returns False."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import lark_oapi  # noqa: F401 - eagerly imported before install_lazy_lark_oapi
from src.utils.lark_lazy import install_lazy_lark_oapi

assert install_lazy_lark_oapi() is False
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_lazy_client_validates_domain():
    """Lazy ClientBuilder enforces the exact same HTTPS/origin rules as SDK ClientBuilder."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
from src.utils.lark_lazy import install_lazy_lark_oapi

install_lazy_lark_oapi()
import lark_oapi as lark

lark.Client.builder().domain("https://open.feishu.cn").app_id("a").app_secret("b").build()
lark.Client.builder().domain("http://127.0.0.1:8080").app_id("a").app_secret("b").build()

for bad in ("", "http://evil.example.com", "https://user:pass@open.feishu.cn", "https://open.feishu.cn/path"):
    try:
        lark.Client.builder().domain(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Expected ValueError for domain={bad!r}")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_lazy_client_request_and_arequest():
    """Lazy Client.request() and Client.arequest() dispatch via SDK Transport."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.lark_lazy import install_lazy_lark_oapi

install_lazy_lark_oapi()
import lark_oapi as lark
from lark_oapi.core.model import BaseRequest, RawResponse

client = lark.Client.builder().app_id("cli_a").app_secret("sec_b").build()
req = BaseRequest()
req.uri = "/open-apis/test"
req.token_types = {lark.AccessTokenType.TENANT}

raw_ok = RawResponse()
raw_ok.status_code = 200
raw_ok.headers = {"Content-Type": "application/json"}
raw_ok.content = b'{"code": 0, "msg": "success"}'

with (
    patch("lark_oapi.core.token.verify") as mock_verify,
    patch("lark_oapi.core.http.Transport.execute", return_value=raw_ok) as mock_exec,
    patch("lark_oapi.core.http.Transport.aexecute", new=AsyncMock(return_value=raw_ok)) as mock_aexec,
):
    resp = client.request(req)
    assert resp.code == 0 and resp.raw is raw_ok
    aresp = asyncio.run(client.arequest(req))
    assert aresp.code == 0 and aresp.raw is raw_ok
    assert mock_verify.call_count == 2
    assert mock_exec.call_count == 1
    assert mock_aexec.call_count == 1
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
