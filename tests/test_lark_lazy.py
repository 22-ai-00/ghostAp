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


def test_install_lazy_lark_channel_lazy_resolution_and_exports():
    """install_lazy_lark_channel() defers channel client/API loading until accessed."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from src.utils.lark_lazy import install_lazy_lark_channel

assert install_lazy_lark_channel() is True
import lark_channel
assert getattr(lark_channel, "_IS_LAZY_LARK_CHANNEL_ROOT", False) is True

from lark_channel import EventDispatcherHandler, LogLevel, TransportConfig
assert EventDispatcherHandler.builder("", "") is not None
assert LogLevel.WARNING is not None
assert TransportConfig(kind="webhook", http_timeout_seconds=5.0).kind == "webhook"

# EventDispatcherHandler / LogLevel / TransportConfig must not load FeishuChannel, client, or cardkit APIs
assert type(sys.modules["lark_channel.channel"]).__name__ == "_LazyChannelSubpackage"
assert "lark_channel.channel.channel" not in sys.modules
assert "lark_channel.client" not in sys.modules
assert not any(m.startswith("lark_channel.api.cardkit") for m in sys.modules)

# Client builder & single-service lazy hydration
client = (
    lark_channel.Client.builder()
    .app_id("cli_a")
    .app_secret("sec_b")
    .proxy_url(None)
    .trust_env_proxy(False)
    .build()
)
assert client.cardkit.v1.card is not None
assert not any(m.startswith("lark_channel.api.drive") for m in sys.modules)
assert not any(m.startswith("lark_channel.api.wiki") for m in sys.modules)

from lark_channel.api.cardkit.v1.model.card import Card
assert Card is not None
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


def test_install_lazy_lark_channel_returns_false_when_real_sdk_loaded_early():
    """If real lark_channel is already in sys.modules, install_lazy_lark_channel returns False."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import lark_channel  # noqa: F401
from src.utils.lark_lazy import install_lazy_lark_channel

assert install_lazy_lark_channel() is False
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


def test_lazy_lark_oapi_root_client_exports_and_version_model_hydration():
    """Lazy root attribute lookup skips core/card until needed and hydrates version models."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from src.utils.lark_lazy import install_lazy_lark_oapi

assert install_lazy_lark_oapi() is True
import lark_oapi as lark

# Accessing a service subpackage directly must not eagerly import core or card
assert lark.im.v1 is not None
assert "lark_oapi.card" not in sys.modules
assert "lark_oapi.core" not in sys.modules

# Standard symbols exported by lark_oapi.client must resolve
from lark_oapi.client import (
    AppType,
    BaseRequest,
    BaseResponse,
    Config,
    Files,
    JSON,
    LogLevel,
    RequestOption,
    TokenManager,
    Transport,
    verify,
)
assert all(
    sym is not None
    for sym in (AppType, BaseRequest, BaseResponse, Config, Files, JSON, LogLevel, RequestOption, TokenManager, Transport, verify)
)

# Importing .version hydrates .model first
import lark_oapi.api.im.v1.version as im_v1_version
from lark_oapi.api.im.v1 import GetChatRequest
assert getattr(im_v1_version, "GetChatRequest", None) is GetChatRequest
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


def test_feishu_ws_client_startup_defers_heavy_sdks_and_engines():
    """Creating FeishuWSClient keeps acp, lark_channel.channel, and heavy engines unloaded."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import sys
import src.main as main_mod

main_mod._activate_lark_import_cache()
main_mod._load_feishu_runtime()
from src.feishu.ws_client import FeishuWSClient

client = FeishuWSClient(message_callback=lambda *_a, **_kw: None)
assert client is not None
for forbidden in (
    "acp",
    "lark_channel.channel",
    "lark_channel.channel.channel",
    "src.workflow_engine.engine",
    "src.spec_engine.engine",
    "src.acp.client",
    "src.acp.session",
):
    assert forbidden not in sys.modules, f"{forbidden} unexpectedly loaded at startup"
assert len([m for m in sys.modules if m.startswith("lark_oapi")]) < 150
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


def test_lazy_httpx_guard_crypto_and_unified_transport():
    """HTTPX CLI deps, Crypto, requests, and requests_toolbelt stay unloaded until needed."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import asyncio
import base64
import hashlib
import sys
from unittest.mock import patch

from src.utils.lark_lazy import (
    install_lazy_httpx_guard,
    install_lazy_lark_channel,
    install_lazy_lark_oapi,
)

assert install_lazy_httpx_guard()
assert install_lazy_lark_oapi()
assert install_lazy_lark_channel()

import httpx
import httpx as httpx_mod
import lark_oapi
import lark_channel
import lark_channel.ws.client as lark_ws_impl
from lark_oapi.core.utils.decryptor import AESCipher as OapiAESCipher
from lark_channel.core.http import Transport as ChannelTransport
from lark_oapi.core.http import Transport as OapiTransport

assert httpx_mod.Client is not None
assert lark_ws_impl.Client is not None

for forbidden in (
    "httpx._main",
    "click",
    "pygments",
    "rich",
    "requests",
    "requests_toolbelt",
    "urllib3",
    "Crypto",
):
    assert forbidden not in sys.modules, f"{forbidden} should not be loaded eagerly"

from lark_oapi.core.enum import HttpMethod as OapiHttpMethod
from lark_oapi.core.model import BaseRequest as OapiBaseRequest, Config as OapiConfig
from lark_channel.core.enum import HttpMethod as ChHttpMethod
from lark_channel.core.model import BaseRequest as ChBaseRequest, Config as ChConfig

# Verify unified httpx Transport.execute & Transport.aexecute without importing requests
req_oapi = OapiBaseRequest.builder().uri("/open-apis/test/:id").paths({"id": "42"}).http_method(OapiHttpMethod.POST).body({"ok": True}).build()
conf_oapi = OapiConfig()
conf_oapi.domain = "https://open.feishu.cn"
conf_oapi.timeout = 5

mock_transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"url": str(request.url)}))
orig_client = httpx.Client
orig_async_client = httpx.AsyncClient

with patch.object(httpx, "Client", lambda **kw: orig_client(transport=mock_transport, **kw)):
    raw = OapiTransport.execute(conf_oapi, req_oapi)
    assert raw.status_code == 200
    assert b"42" in raw.content
assert "requests" not in sys.modules

req_ch = ChBaseRequest.builder().uri("/open-apis/ch").http_method(ChHttpMethod.POST).body({"hello": "world"}).build()
conf_ch = ChConfig()
conf_ch.domain = "https://open.feishu.cn"
conf_ch.timeout = 5
with patch.object(httpx, "AsyncClient", lambda **kw: orig_async_client(transport=mock_transport, **kw)):
    raw_async = asyncio.run(ChannelTransport.aexecute(conf_ch, req_ch))
    assert raw_async.status_code == 200
assert "requests" not in sys.modules
assert "Crypto" not in sys.modules

# Verify lazy decryptor still hydrates Crypto on first decrypt_str call
from Crypto.Cipher import AES
key = "secret-key"
digest = hashlib.sha256(key.encode("utf-8")).digest()
iv = b"0123456789abcdef"
plaintext = b"ghostap-secret"
pad = 16 - (len(plaintext) % 16)
ct = iv + AES.new(digest, AES.MODE_CBC, iv).encrypt(plaintext + bytes([pad]) * pad)
enc = base64.b64encode(ct).decode("ascii")
assert OapiAESCipher(key).decrypt_str(enc) == "ghostap-secret"
assert "Crypto" in sys.modules
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
