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


def test_feishu_ws_client_startup_and_cold_hydration_dispatch(tmp_path):
    """Startup keeps heavy subtrees unloaded and cold hydration works cleanly on demand."""
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import sys
from unittest.mock import MagicMock, patch

import src.main as main_mod

main_mod._activate_lark_import_cache()
main_mod._load_feishu_runtime()
from src.feishu.ws_client import FeishuWSClient

client = FeishuWSClient(message_callback=lambda *_a, **_kw: None)

# 1. Verify heavy modules remain unloaded after cold startup init
for forbidden in (
    "acp",
    "src.acp.client",
    "src.acp.session",
    "src.acp.sync_adapter",
    "src.acp.helper",
    "src.acp.continuation",
    "src.agent_session.claude_cli",
    "src.agent_session.factory",
    "src.card.session",
    "src.card.session.core",
    "src.card.render",
    "src.card.delivery",
    "src.card.events",
    "src.card.builders",
    "src.card.models",
    "src.deep_engine.engine",
    "src.spec_engine.engine",
    "src.spec_engine.models",
    "src.workflow_engine.engine",
    "src.workflow_engine.renderer",
    "lark_channel",
    "lark_channel.ws",
    "lark_channel.channel",
    "psutil",
    "requests",
    "requests_toolbelt",
    "urllib3",
    "Crypto",
    "click",
    "pygments",
    "rich",
    "websockets",
):
    assert forbidden not in sys.modules, f"{{forbidden}} unexpectedly loaded at startup"

assert len(sys.modules) < 520, f"too many startup modules: {{len(sys.modules)}}"
src_mods = [m for m in sys.modules if m == "src" or m.startswith("src.")]
assert len(src_mods) <= 130, f"too many src modules at startup: {{len(src_mods)}}"

# 2. Hydrate lazy handlers, renderers, engines, IM client, card callback, and GCMonitor
handlers = client._handler_ctx.handlers
for h in handlers.values():
    h.reply_card = MagicMock(return_value="om_mock_reply")
    h.reply_interactive_card = MagicMock(return_value="om_mock_reply")
    h.reply_text = MagicMock(return_value="om_mock_text")
    h.reply_error = MagicMock(return_value="om_mock_err")

# Deep & Spec lazy renderers + engine creation
deep_renderer = handlers["deep"].renderer
spec_renderer = handlers["spec"].renderer
assert deep_renderer is not None
assert spec_renderer is not None
assert "src.deep_engine.engine" in sys.modules

# SpecReporter lazy models hydration + SpecEngineManager cold engine creation
spec_eng = client._spec_engine_manager.get_or_create("oc_cold", {str(tmp_path)!r})
assert spec_eng is not None
assert "src.spec_engine.engine" in sys.modules
spec_status = client._spec_reporter.format_phase_progress(
    __import__("src.spec_engine.models", fromlist=["SpecPhase"]).SpecPhase.SPEC
)
assert spec_status

# Workflow lazy storage helper + engine manager
from src.feishu.handlers.workflow_execution import _workflow_module
assert isinstance(_workflow_module().workflow_scripts_dir({str(tmp_path)!r}), str)

# System, Project, Diagnostics lazy card builder hydration
handlers["system"].handle_menu_command("om_1", "oc_cold")
handlers["system"].show_full_help("om_2", "oc_cold")
handlers["project"].show_project_board("om_3", "oc_cold")
handlers["diagnostics"].show_task_board("om_4", "oc_cold", "/tasks")
assert handlers["system"].reply_card.called
assert handlers["project"].reply_card.called

# Card callback 2.0 response hydration
from src.feishu.card_callback_contract import build_card_action_response
resp = build_card_action_response(
    toast_type="info",
    toast_content="hydrated",
    card={{"schema": "2.0", "body": {{"elements": [{{"tag": "markdown", "content": "ok"}}]}}}},
)
assert resp is not None

# FeishuIMClient lazy im.v1 model hydration
from src.feishu.im_client import FeishuIMClient
mock_api = MagicMock()
mock_api.im.v1.message.reply.return_value = MagicMock(success=lambda: True)
im_client = FeishuIMClient(lambda: mock_api, client.settings)
assert im_client.reply_message("om_1", '{{"text":"hi"}}', msg_type="text") is not None

# GCMonitor lazy psutil hydration
from src.utils.gc_monitor import GCMonitor
with patch("gc.collect"):
    GCMonitor(memory_threshold_percent=99.9, check_interval_seconds=0.0).check_and_collect("Test")
assert "psutil" in sys.modules
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


def test_concurrent_sync_main_bot_identity_never_hits_lark_core_circular_import():
    """Concurrent callers entering _sync_main_bot_identity hydrate lark_oapi.core safely."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import src.main as main_mod
main_mod._activate_lark_import_cache()
main_mod._load_feishu_runtime()
from src.feishu.ws_client import FeishuWSClient

client = FeishuWSClient(message_callback=lambda *_a, **_kw: None)
mock_api = MagicMock()
mock_api.request.return_value = SimpleNamespace(
    success=lambda: True,
    raw=None,
    data=None,
)
client._get_api_client = lambda: mock_api

barrier = threading.Barrier(4)
errors = []

def _worker():
    barrier.wait()
    try:
        client._sync_main_bot_identity()
    except Exception as exc:
        errors.append(repr(exc))

threads = [threading.Thread(target=_worker) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=5)
assert not errors, f"concurrent _sync_main_bot_identity failed: {errors}"
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


def test_ws_event_handler_and_log_level_avoid_loading_feishu_channel():
    """Accessing WS dispatcher/log level must not hydrate FeishuChannel or Drive/Wiki APIs."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import sys
import src.main as main_mod
main_mod._activate_lark_import_cache()
main_mod._load_feishu_runtime()

import src.feishu.ws_client as ws_mod
from src.feishu.ws_client import FeishuWSClient

client = FeishuWSClient(message_callback=lambda *_a, **_kw: None)
try:
    handler = client._build_event_handler()
    log_level = getattr(ws_mod, "ChannelLogLevel")
    assert handler is not None
    assert log_level is not None
    assert "FeishuChannel" not in ws_mod.__dict__
    assert "lark_channel.channel.channel" not in sys.modules
    assert "lark_channel.api.drive" not in sys.modules
    assert "lark_channel.api.wiki" not in sys.modules
    lc_mods = [m for m in sys.modules if m == "lark_channel" or m.startswith("lark_channel.")]
    assert len(lc_mods) <= 90, f"too many lark_channel modules loaded: {len(lc_mods)}"
finally:
    client.close()
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


def test_concurrent_lazy_sdk_hydration_across_threads():
    """Concurrent threads touching lazy SDK trees hydrate core safely without circular import errors."""
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import threading
import src.main as main_mod
main_mod._activate_lark_import_cache()
main_mod._load_feishu_runtime()

import lark_oapi as lark
import lark_channel as lc
from src.feishu.ws_client import FeishuWSClient

ws = FeishuWSClient(message_callback=lambda *_a, **_kw: None)
try:
    for _ in range(3):
        barrier = threading.Barrier(8)
        errors = []

        def run(fn):
            barrier.wait()
            try:
                fn()
            except Exception as exc:
                errors.append(repr(exc))

        def check_client():
            assert lark.Client.builder().app_id("a").app_secret("b").build() is not None

        def check_im_req():
            from lark_oapi.api.im.v1 import CreateMessageRequest
            assert CreateMessageRequest is not None

        def check_transport():
            from lark_oapi.core.http import Transport
            assert Transport is not None

        def check_resource():
            c = lark.Client.builder().app_id("a").app_secret("b").build()
            assert c.im.v1.message is not None

        def check_channel_client():
            assert lc.Client.builder().app_id("a").app_secret("b").build() is not None

        def check_feishu_channel():
            from lark_channel import FeishuChannel
            assert FeishuChannel is not None

        def check_cardkit():
            from lark_oapi.api.cardkit.v1 import CreateCardRequest
            assert CreateCardRequest is not None

        def check_ws_api_client():
            assert ws._get_api_client() is not None

        threads = [
            threading.Thread(target=run, args=(fn,))
            for fn in (
                check_client,
                check_im_req,
                check_transport,
                check_resource,
                check_channel_client,
                check_feishu_channel,
                check_cardkit,
                check_ws_api_client,
            )
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not errors, f"concurrent lazy hydration errors: {errors}"
finally:
    ws.close()
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


def test_httpx_transport_cached_ssl_multipart_async_and_ws_compat(tmp_path):
    """Verify SSLContext caching, DEBUG None queries, async multipart upload, retry, and WS requests compat."""
    repo_root = Path(__file__).resolve().parents[1]
    payload_file = tmp_path / "retry_upload.txt"
    payload_file.write_bytes(b"hello-retry-payload")
    script = f"""
import asyncio
import io
import sys
from types import SimpleNamespace
from unittest.mock import patch
import httpx

from src.utils.lark_lazy import (
    _httpx_client_kwargs_for,
    install_lazy_lark_channel,
    install_lazy_lark_oapi,
)

install_lazy_lark_oapi()
install_lazy_lark_channel()

import lark_oapi as lark
from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody
from src.feishu.im_client import FeishuIMClient

captured = []
file_calls = 0
observed_compat = []

def handler(request: httpx.Request) -> httpx.Response:
    global file_calls
    captured.append((request.method, str(request.url), request.read()))
    if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
        return httpx.Response(200, json={{"code": 0, "tenant_access_token": "t-tok", "expire": 7200}})
    if request.url.path == "/open-apis/im/v1/files":
        file_calls += 1
        if file_calls == 1:
            return httpx.Response(500, json={{"code": 999, "msg": "retry me"}})
        return httpx.Response(200, json={{"code": 0, "data": {{"file_key": "file_retry_ok"}}}})
    if request.url.path == "/open-apis/im/v1/images":
        return httpx.Response(200, json={{"code": 0, "data": {{"image_key": "img_ok"}}}})
    if request.url.path == "/ws/endpoint":
        return httpx.Response(200, json={{"code": 0, "data": {{"URL": "wss://example.invalid"}}}})
    return httpx.Response(404)

transport = httpx.MockTransport(handler)
orig_client = httpx.Client
orig_async_client = httpx.AsyncClient

def patched_client(**kw):
    observed_compat.append(kw)
    return orig_client(transport=transport, **{{k: v for k, v in kw.items() if k != "transport"}})

def patched_async_client(**kw):
    return orig_async_client(transport=transport, **{{k: v for k, v in kw.items() if k != "transport"}})

client = (
    lark.Client.builder()
    .app_id("cli_x")
    .app_secret("sec_x")
    .domain("https://open.feishu.cn")
    .log_level(lark.LogLevel.DEBUG)
    .build()
)

# Real httpx.Client gets cached SSLContext; mocked clients do not
kw1 = _httpx_client_kwargs_for(client.config, orig_client, httpx_mod=httpx)
kw2 = _httpx_client_kwargs_for(client.config, orig_client, httpx_mod=httpx)
assert kw1.get("verify") is not None and kw1.get("verify") is kw2.get("verify")
fake_cls = type("FakeClient", (), {{"__module__": "tests.fake"}})
assert "verify" not in _httpx_client_kwargs_for(client.config, fake_cls)

with patch("httpx.Client", side_effect=patched_client), patch(
    "httpx.AsyncClient", side_effect=patched_async_client
), patch("time.sleep"):
    from lark_oapi.ws import client as oapi_ws
    from lark_channel.ws import client as chan_ws

    r1 = oapi_ws.requests.post(
        "https://open.feishu.cn/ws/endpoint",
        json={{"a": 1}},
        allow_redirects=False,
        verify=False,
    )
    r2 = chan_ws.requests.Session().post("https://open.feishu.cn/ws/endpoint", json={{"b": 2}})
    assert r1.status_code == 200 and r2.status_code == 200
    assert observed_compat[0].get("verify") is False
    assert "requests" not in sys.modules

    im = FeishuIMClient(
        lambda: client,
        SimpleNamespace(im_api_max_retries=2, im_api_retry_delay_s=0.01),
    )
    assert im.upload_file({str(payload_file)!r}, file_type="stream", file_name="hello.txt") == "file_retry_ok"
    assert im.upload_image_bytes(b"PNGDATA") == "img_ok"

    req_async = CreateImageRequest.builder().request_body(
        CreateImageRequestBody.builder().image_type("message").image(io.BytesIO(b"ASYNCIMG")).build()
    ).build()
    aresp = asyncio.run(client.im.v1.image.acreate(req_async))
    assert aresp.success() and aresp.data.image_key == "img_ok"

file_reqs = [c for c in captured if "/open-apis/im/v1/files" in c[1]]
assert len(file_reqs) == 2
assert b"hello-retry-payload" in file_reqs[0][2]
assert b"hello-retry-payload" in file_reqs[1][2]
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
