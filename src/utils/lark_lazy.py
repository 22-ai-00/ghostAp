"""Lazy import & Client proxy for ``lark_oapi``.

``lark_oapi``'s generated top-level ``__init__.py`` executes ``from .api import *``
and imports ``Client``, eagerly loading 56 services and 10,600+ model files
(~165–265 MB RSS, ~2.5s+ startup CPU). GhostAP uses only a handful of endpoints
from ``im.v1``, ``cardkit.v1``, ``contact.v3``, and ``application.v7``.

Calling :func:`install_lazy_lark_oapi` before ``lark_oapi`` is imported installs
a ``MetaPathFinder`` that:

1. Replaces ``lark_oapi``, ``lark_oapi.api``, ``lark_oapi.api.<svc>``,
   ``lark_oapi.api.<svc>.<ver>``, and ``lark_oapi.api.<svc>.<ver>.model`` with
   lazy package modules that resolve specific model files on demand (with exact
   fallback to the real ``__init__.py`` if a symbol uses non-standard naming).
2. Replaces ``lark_oapi.client`` with a drop-in ``Client`` / ``ClientBuilder``
   whose service/version/resource chain (e.g. ``client.im.v1.message``)
   imports and instantiates only the specific resource classes actually used.
"""

from __future__ import annotations

import ast
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import logging
import re
import sys
import threading
import types
import urllib.parse
from pathlib import Path
from typing import Any, Optional

_INSTALL_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_BLOCKED_HTTPX_SUBMODULES = frozenset({"httpx._main"})

_TOP_LEVEL_SUBMODULES = frozenset(
    {"api", "card", "channel", "client", "core", "event", "scene", "ws"}
)
_TOP_LEVEL_SYMBOL_MODULES = {
    "EventContext": ("lark_oapi.event.context", "EventContext"),
    "CustomizedEvent": ("lark_oapi.event.custom", "CustomizedEvent"),
    "EventDispatcherHandler": ("lark_oapi.event.dispatcher_handler", "EventDispatcherHandler"),
    "register_app": ("lark_oapi.scene.registration", "register_app"),
    "aregister_app": ("lark_oapi.scene.registration", "aregister_app"),
    "CardActionHandler": ("lark_oapi.card.action_handler", "CardActionHandler"),
}

_CHANNEL_TOP_SUBMODULES = frozenset(
    {"api", "channel", "client", "core", "event", "ws"}
)
_CHANNEL_TOP_SYMBOL_MODULES = {
    "EventContext": ("lark_channel.event.context", "EventContext"),
    "CustomizedEvent": ("lark_channel.event.custom", "CustomizedEvent"),
    "EventDispatcherHandler": ("lark_channel.event.dispatcher_handler", "EventDispatcherHandler"),
    "FeishuChannel": ("lark_channel.channel.channel", "FeishuChannel"),
    "TransportConfig": ("lark_channel.channel.config", "TransportConfig"),
    "OutboundCard": ("lark_channel.channel.types", "OutboundCard"),
    "FeishuChannelErrorCode": ("lark_channel.channel.errors", "FeishuChannelErrorCode"),
}
_CHANNEL_SUBPKG_SYMBOL_MODULES = {
    "FeishuChannel": "lark_channel.channel.channel",
    "TransportConfig": "lark_channel.channel.config",
    "ChannelConfig": "lark_channel.channel.config",
    "SafetyConfig": "lark_channel.channel.config",
    "SecurityConfig": "lark_channel.channel.config",
    "PolicyConfig": "lark_channel.channel.config",
    "InboundConfig": "lark_channel.channel.config",
    "OutboundConfig": "lark_channel.channel.config",
    "KeepaliveConfig": "lark_channel.channel.config",
    "MediaCacheConfig": "lark_channel.channel.config",
    "ChatModeCacheConfig": "lark_channel.channel.config",
    "ChatQueueConfig": "lark_channel.channel.config",
    "DedupConfig": "lark_channel.channel.config",
    "TextBatchConfig": "lark_channel.channel.config",
    "RetryConfig": "lark_channel.channel.config",
    "UATConfig": "lark_channel.channel.config",
    "OutboundCard": "lark_channel.channel.types",
    "FeishuChannelError": "lark_channel.channel.errors",
    "FeishuChannelErrorCode": "lark_channel.channel.errors",
    "OutboundSendError": "lark_channel.channel.errors",
    "SendError": "lark_channel.channel.errors",
    "ChannelEventName": "lark_channel.channel.events",
    "Events": "lark_channel.channel.events",
}


def _to_snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _model_module_candidates(symbol: str) -> list[str]:
    """Return ordered snake_case module name candidates for a model symbol."""
    seen: list[str] = []

    def _add(val: str) -> None:
        if val and val not in seen:
            seen.append(val)

    _add(_to_snake_case(symbol))
    if symbol.endswith("Builder") and len(symbol) > 7:
        _add(_to_snake_case(symbol[:-7]))
    if symbol.endswith("Data") and len(symbol) > 4:
        _add(_to_snake_case(symbol[:-4]))
    if symbol.startswith("P2"):
        for suffix in ("Event", "Action", "Body"):
            if symbol.endswith(suffix) and len(symbol) > len(suffix):
                _add(_to_snake_case(symbol[: -len(suffix)]))
    return seen


def _exec_real_init(module: types.ModuleType, init_file: Path) -> None:
    """Fallback: execute a package's real ``__init__.py`` into ``module``."""
    if not init_file.is_file():
        return
    if getattr(module, "_lazy_real_init_loaded", False):
        return
    setattr(module, "_lazy_real_init_loaded", True)
    code = compile(init_file.read_text(encoding="utf-8"), str(init_file), "exec")
    exec(code, module.__dict__)


class _LazyLarkPackage(types.ModuleType):
    """Base lazy ``ModuleType`` with directory-backed submodule & symbol lookup."""

    def __init__(self, fullname: str, pkg_dir: Path) -> None:
        super().__init__(fullname)
        self.__file__ = str(pkg_dir / "__init__.py")
        self.__path__ = [str(pkg_dir)]
        self.__package__ = fullname
        self._lazy_pkg_dir = pkg_dir
        self._lazy_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock


class _LazyLarkRootModule(_LazyLarkPackage):
    """Lazy replacement for top-level ``lark_oapi``."""

    _IS_LAZY_LARK_ROOT = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"module 'lark_oapi' has no attribute {name!r}")

        with self._lazy_lock:
            if name in self.__dict__:
                return self.__dict__[name]

            if name == "Client":
                client_mod = importlib.import_module("lark_oapi.client")
                val = client_mod.Client
                self.__dict__["Client"] = val
                return val

            target = _TOP_LEVEL_SYMBOL_MODULES.get(name)
            if target is not None:
                mod = importlib.import_module(target[0])
                val = getattr(mod, target[1])
                self.__dict__[name] = val
                return val

            pkg_dir = self._lazy_pkg_dir
            if (
                name in _TOP_LEVEL_SUBMODULES
                or (pkg_dir / name).is_dir()
                or (pkg_dir / f"{name}.py").is_file()
            ):
                mod = importlib.import_module(f"lark_oapi.{name}")
                self.__dict__[name] = mod
                return mod

            # Check `lark_oapi.api.<svc>` before eagerly loading `core` or `card`
            if (pkg_dir / "api" / name).is_dir():
                mod = importlib.import_module(f"lark_oapi.api.{name}")
                self.__dict__[name] = mod
                return mod

            # Exports from `lark_oapi.core` (`from .core import *`)
            core_mod = importlib.import_module("lark_oapi.core")
            if hasattr(core_mod, name):
                val = getattr(core_mod, name)
                self.__dict__[name] = val
                return val

            # Exports from `lark_oapi.card.model` (`from .card import *`)
            card_mod = importlib.import_module("lark_oapi.card")
            if hasattr(card_mod, name):
                val = getattr(card_mod, name)
                self.__dict__[name] = val
                return val

            # Exports from `lark_oapi.api` (`from .api import *`)
            api_mod = importlib.import_module("lark_oapi.api")
            if hasattr(api_mod, name):
                val = getattr(api_mod, name)
                self.__dict__[name] = val
                return val

        raise AttributeError(f"module 'lark_oapi' has no attribute {name!r}")


class _LazyChannelRootModule(_LazyLarkPackage):
    """Lazy replacement for top-level ``lark_channel``."""

    _IS_LAZY_LARK_CHANNEL_ROOT = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"module 'lark_channel' has no attribute {name!r}")

        with self._lazy_lock:
            if name in self.__dict__:
                return self.__dict__[name]

            if name == "Client":
                client_mod = importlib.import_module("lark_channel.client")
                val = client_mod.Client
                self.__dict__["Client"] = val
                return val

            target = _CHANNEL_TOP_SYMBOL_MODULES.get(name)
            if target is not None:
                mod = importlib.import_module(target[0])
                val = getattr(mod, target[1])
                self.__dict__[name] = val
                return val

            pkg_dir = self._lazy_pkg_dir
            if (
                name in _CHANNEL_TOP_SUBMODULES
                or (pkg_dir / name).is_dir()
                or (pkg_dir / f"{name}.py").is_file()
            ):
                mod = importlib.import_module(f"lark_channel.{name}")
                self.__dict__[name] = mod
                return mod

            core_mod = importlib.import_module("lark_channel.core")
            if hasattr(core_mod, name):
                val = getattr(core_mod, name)
                self.__dict__[name] = val
                return val

            for submod_name in (
                "lark_channel.channel.config",
                "lark_channel.channel.errors",
                "lark_channel.channel.events",
                "lark_channel.channel.types",
            ):
                submod = importlib.import_module(submod_name)
                if hasattr(submod, name):
                    val = getattr(submod, name)
                    self.__dict__[name] = val
                    return val

            channel_mod = importlib.import_module("lark_channel.channel")
            if hasattr(channel_mod, name):
                val = getattr(channel_mod, name)
                self.__dict__[name] = val
                return val

        raise AttributeError(f"module 'lark_channel' has no attribute {name!r}")


class _LazyChannelSubpackage(_LazyLarkPackage):
    """Lazy package for ``lark_channel.channel`` avoiding eager full-tree load."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        with self._lazy_lock:
            if name in self.__dict__:
                return self.__dict__[name]

            target_mod_name = _CHANNEL_SUBPKG_SYMBOL_MODULES.get(name)
            if target_mod_name is not None:
                mod = importlib.import_module(target_mod_name)
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    self.__dict__[name] = val
                    return val

            pkg_dir = self._lazy_pkg_dir
            if (pkg_dir / name).is_dir() or (pkg_dir / f"{name}.py").is_file():
                mod = importlib.import_module(f"{self.__name__}.{name}")
                self.__dict__[name] = mod
                return mod

            for submod_name in (
                "lark_channel.channel.config",
                "lark_channel.channel.errors",
                "lark_channel.channel.events",
                "lark_channel.channel.types",
            ):
                submod = importlib.import_module(submod_name)
                if hasattr(submod, name):
                    val = getattr(submod, name)
                    self.__dict__[name] = val
                    return val

            _exec_real_init(self, pkg_dir / "__init__.py")
            if name in self.__dict__:
                return self.__dict__[name]

        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


class _LazySubmodulePackage(_LazyLarkPackage):
    """Lazy package for ``lark_oapi.api`` and ``lark_oapi.api.<svc>``."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        with self._lazy_lock:
            if name in self.__dict__:
                return self.__dict__[name]

            pkg_dir = self._lazy_pkg_dir
            if (pkg_dir / name).is_dir() or (pkg_dir / f"{name}.py").is_file():
                mod = importlib.import_module(f"{self.__name__}.{name}")
                self.__dict__[name] = mod
                return mod

        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


class _LazyModelPackage(_LazyLarkPackage):
    """Lazy package for ``lark_oapi.api.<svc>.<ver>.model``."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        with self._lazy_lock:
            if name in self.__dict__:
                return self.__dict__[name]

            pkg_dir = self._lazy_pkg_dir
            if (pkg_dir / f"{name}.py").is_file():
                mod = importlib.import_module(f"{self.__name__}.{name}")
                self.__dict__[name] = mod
                return mod

            for cand in _model_module_candidates(name):
                if (pkg_dir / f"{cand}.py").is_file():
                    submod = importlib.import_module(f"{self.__name__}.{cand}")
                    if hasattr(submod, name):
                        val = getattr(submod, name)
                        self.__dict__[name] = val
                        return val

            _exec_real_init(self, pkg_dir / "__init__.py")
            if name in self.__dict__:
                return self.__dict__[name]

        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


class _LazyVersionPackage(_LazyLarkPackage):
    """Lazy package for ``lark_oapi.api.<svc>.<ver>``."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        with self._lazy_lock:
            if name in self.__dict__:
                return self.__dict__[name]

            pkg_dir = self._lazy_pkg_dir
            if (pkg_dir / name).is_dir() or (pkg_dir / f"{name}.py").is_file():
                mod = importlib.import_module(f"{self.__name__}.{name}")
                self.__dict__[name] = mod
                return mod

            if re.fullmatch(r"V\d+", name) and (pkg_dir / "version.py").is_file():
                ver_mod = importlib.import_module(f"{self.__name__}.version")
                if hasattr(ver_mod, name):
                    val = getattr(ver_mod, name)
                    self.__dict__[name] = val
                    return val

            model_dir = pkg_dir / "model"
            if model_dir.is_dir():
                for cand in _model_module_candidates(name):
                    if (model_dir / f"{cand}.py").is_file():
                        submod = importlib.import_module(f"{self.__name__}.model.{cand}")
                        if hasattr(submod, name):
                            val = getattr(submod, name)
                            self.__dict__[name] = val
                            return val

            if model_dir.is_dir():
                model_pkg = importlib.import_module(f"{self.__name__}.model")
                _exec_real_init(model_pkg, model_dir / "__init__.py")
                for k, v in vars(model_pkg).items():
                    if not k.startswith("_") and k not in self.__dict__:
                        self.__dict__[k] = v
            _exec_real_init(self, pkg_dir / "__init__.py")
            if name in self.__dict__:
                return self.__dict__[name]

        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


class _LazyVersionProxy:
    """Lazy proxy for ``client.<svc>.<ver>`` (loads only accessed resources)."""

    def __init__(
        self,
        config: Any,
        svc_name: str,
        ver_name: str,
        ver_dir: Path,
        *,
        pkg_prefix: str = "lark_oapi",
    ) -> None:
        self._config = config
        self._svc_name = svc_name
        self._ver_name = ver_name
        self._ver_dir = ver_dir
        self._pkg_prefix = pkg_prefix
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._fallback_version: Any = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        with self._lock:
            if name in self.__dict__:
                return self.__dict__[name]

            res_file = self._ver_dir / "resource" / f"{name}.py"
            if res_file.is_file():
                mod_name = f"{self._pkg_prefix}.api.{self._svc_name}.{self._ver_name}.resource.{name}"
                res_mod = importlib.import_module(mod_name)
                expected_cls = "".join(part.capitalize() for part in name.split("_"))
                cls = getattr(res_mod, expected_cls, None)
                if not isinstance(cls, type):
                    candidates = [
                        obj
                        for obj in vars(res_mod).values()
                        if isinstance(obj, type) and obj.__module__ == mod_name
                    ]
                    if len(candidates) == 1:
                        cls = candidates[0]
                    else:
                        for cand in candidates:
                            if cand.__name__.lower() == expected_cls.lower():
                                cls = cand
                                break
                if isinstance(cls, type):
                    inst = cls(self._config)
                    self.__dict__[name] = inst
                    return inst

            # Fallback to real Version class only if `version.py` defines `name`
            version_file = self._ver_dir / "version.py"
            if version_file.is_file():
                if self._fallback_version is None:
                    try:
                        version_src = version_file.read_text(encoding="utf-8")
                    except OSError:
                        version_src = ""
                    if f"self.{name}" not in version_src and f"def {name}" not in version_src:
                        raise AttributeError(
                            f"'{self._svc_name}.{self._ver_name}' has no resource {name!r}"
                        )
                    res_pkg = importlib.import_module(
                        f"{self._pkg_prefix}.api.{self._svc_name}.{self._ver_name}.resource"
                    )
                    _exec_real_init(res_pkg, self._ver_dir / "resource" / "__init__.py")
                    ver_mod = importlib.import_module(
                        f"{self._pkg_prefix}.api.{self._svc_name}.{self._ver_name}.version"
                    )
                    cls_name = self._ver_name.capitalize()
                    ver_cls = getattr(ver_mod, cls_name, None)
                    if isinstance(ver_cls, type):
                        self._fallback_version = ver_cls(self._config)
                if self._fallback_version is not None and hasattr(self._fallback_version, name):
                    val = getattr(self._fallback_version, name)
                    self.__dict__[name] = val
                    return val

        raise AttributeError(
            f"'{self._svc_name}.{self._ver_name}' has no resource {name!r}"
        )


class _LazyServiceProxy:
    """Lazy proxy for ``client.<svc>`` (loads only accessed API versions)."""

    def __init__(
        self,
        config: Any,
        svc_name: str,
        svc_dir: Path,
        *,
        pkg_prefix: str = "lark_oapi",
    ) -> None:
        self._config = config
        self._svc_name = svc_name
        self._svc_dir = svc_dir
        self._pkg_prefix = pkg_prefix
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._fallback_service: Any = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        with self._lock:
            if name in self.__dict__:
                return self.__dict__[name]

            ver_dir = self._svc_dir / name
            if ver_dir.is_dir() and re.fullmatch(r"v\d+", name):
                proxy = _LazyVersionProxy(
                    self._config,
                    self._svc_name,
                    name,
                    ver_dir,
                    pkg_prefix=self._pkg_prefix,
                )
                self.__dict__[name] = proxy
                return proxy

            service_file = self._svc_dir / "service.py"
            if service_file.is_file():
                if self._fallback_service is None:
                    try:
                        svc_src = service_file.read_text(encoding="utf-8")
                    except OSError:
                        svc_src = ""
                    if f"self.{name}" not in svc_src and f"def {name}" not in svc_src:
                        raise AttributeError(
                            f"service '{self._svc_name}' has no attribute {name!r}"
                        )
                    svc_mod = importlib.import_module(
                        f"{self._pkg_prefix}.api.{self._svc_name}.service"
                    )
                    for obj in vars(svc_mod).values():
                        if isinstance(obj, type) and obj.__module__ == svc_mod.__name__:
                            self._fallback_service = obj(self._config)
                            break
                if self._fallback_service is not None and hasattr(self._fallback_service, name):
                    val = getattr(self._fallback_service, name)
                    self.__dict__[name] = val
                    return val

        raise AttributeError(f"service '{self._svc_name}' has no attribute {name!r}")


def _build_lazy_client_module(
    pkg_root: Path, *, pkg_prefix: str = "lark_oapi"
) -> types.ModuleType:
    """Construct the lazy drop-in replacement for ``<pkg_prefix>.client``."""
    core_mod = importlib.import_module(f"{pkg_prefix}.core")
    JSON = core_mod.JSON
    logger = core_mod.logger
    _core_http = importlib.import_module(f"{pkg_prefix}.core.http")
    _core_token = importlib.import_module(f"{pkg_prefix}.core.token")
    TokenManager = _core_token.TokenManager
    ICache = importlib.import_module(f"{pkg_prefix}.core.cache").ICache
    const_mod = importlib.import_module(f"{pkg_prefix}.core.const")
    APPLICATION_JSON = const_mod.APPLICATION_JSON
    CONTENT_TYPE = const_mod.CONTENT_TYPE
    UTF_8 = const_mod.UTF_8
    enum_mod = importlib.import_module(f"{pkg_prefix}.core.enum")
    AppType = enum_mod.AppType
    LogLevel = enum_mod.LogLevel
    model_mod = importlib.import_module(f"{pkg_prefix}.core.model")
    BaseRequest = model_mod.BaseRequest
    BaseResponse = model_mod.BaseResponse
    Config = model_mod.Config
    RequestOption = model_mod.RequestOption
    Files = importlib.import_module(f"{pkg_prefix}.core.utils.files").Files
    if pkg_prefix == "lark_oapi":
        AccessToken: Any = importlib.import_module(f"{pkg_prefix}.core.access_token").AccessToken
    else:
        AccessToken = None

    api_root = pkg_root / "api"

    def _validate_domain(domain: str) -> None:
        if not isinstance(domain, str) or not domain:
            raise ValueError("domain must be a non-empty https URL")
        parsed = urllib.parse.urlsplit(domain)
        if parsed.scheme == "http":
            host = (parsed.hostname or "").lower()
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError(f"domain must use https scheme (got {domain!r})")
        elif parsed.scheme != "https":
            raise ValueError(f"domain must use https scheme (got {domain!r})")
        if not parsed.netloc:
            raise ValueError(f"domain missing host (got {domain!r})")
        if "@" in parsed.netloc:
            raise ValueError(f"domain must not embed credentials (got {domain!r})")
        if parsed.path or parsed.query or parsed.fragment:
            raise ValueError(
                f"domain must be an origin (scheme + host[:port]) without path/query/fragment (got {domain!r})"
            )

    class Client:
        """Drop-in lazy replacement for ``Client``."""

        def __init__(self) -> None:
            self._config: Optional[Config] = None
            if AccessToken is not None:
                self.access_token: Any = None
            self._lazy_services_enabled: bool = False
            self._lazy_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

        @staticmethod
        def builder() -> "ClientBuilder":
            return ClientBuilder()

        @property
        def config(self) -> Optional[Config]:
            """Read-only view of the client's ``Config``."""
            return self._config

        def __getattr__(self, name: str) -> Any:
            if name.startswith("_"):
                raise AttributeError(name)
            svc_dir = api_root / name
            if svc_dir.is_dir():
                if not self._lazy_services_enabled:
                    return None
                with self._lazy_lock:
                    if name in self.__dict__:
                        return self.__dict__[name]
                    proxy = _LazyServiceProxy(
                        self._config, name, svc_dir, pkg_prefix=pkg_prefix
                    )
                    self.__dict__[name] = proxy
                    return proxy
            raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

        def request(
            self, request: BaseRequest, option: Optional[RequestOption] = None
        ) -> BaseResponse:
            if option is None:
                option = RequestOption()
            _core_token.verify(self._config, request, option)
            raw_resp = _core_http.Transport.execute(self._config, request, option)
            resp = BaseResponse()
            content_type = raw_resp.headers.get(CONTENT_TYPE)
            if content_type is not None and content_type.startswith(APPLICATION_JSON):
                resp = JSON.unmarshal(str(raw_resp.content, UTF_8), BaseResponse)
            elif 200 <= raw_resp.status_code < 300:
                resp.code = 0
            resp.raw = raw_resp
            return resp

        async def arequest(
            self, request: BaseRequest, option: Optional[RequestOption] = None
        ) -> BaseResponse:
            if option is None:
                option = RequestOption()
            _core_token.verify(self._config, request, option)
            request.files = Files.extract_files(request.body)
            raw_resp = await _core_http.Transport.aexecute(self._config, request, option)
            resp = BaseResponse()
            content_type = raw_resp.headers.get(CONTENT_TYPE)
            if content_type is not None and content_type.startswith(APPLICATION_JSON):
                resp = JSON.unmarshal(str(raw_resp.content, UTF_8), BaseResponse)
            elif 200 <= raw_resp.status_code < 300:
                resp.code = 0
            resp.raw = raw_resp
            return resp

    class ClientBuilder:
        """Drop-in replacement for ``ClientBuilder``."""

        def __init__(self) -> None:
            self._config = Config()

        def app_id(self, app_id: str) -> "ClientBuilder":
            self._config.app_id = app_id
            return self

        def app_secret(self, app_secret: str) -> "ClientBuilder":
            self._config.app_secret = app_secret
            return self

        if pkg_prefix == "lark_oapi":
            def client_assertion_provider(self, provider: Any) -> "ClientBuilder":
                self._config.client_assertion_provider = provider
                return self

            def oauth_base_url(self, oauth_base_url: str) -> "ClientBuilder":
                self._config.oauth_base_url = oauth_base_url
                return self
        else:
            def proxy_url(self, proxy_url: Optional[str]) -> "ClientBuilder":
                self._config.proxy_url = proxy_url
                return self

            def trust_env_proxy(self, trust_env_proxy: Optional[bool]) -> "ClientBuilder":
                self._config.trust_env_proxy = trust_env_proxy
                return self

        def domain(self, domain: str) -> "ClientBuilder":
            _validate_domain(domain)
            self._config.domain = domain
            return self

        def timeout(self, timeout: float) -> "ClientBuilder":
            self._config.timeout = timeout
            return self

        def app_type(self, app_type: AppType) -> "ClientBuilder":
            self._config.app_type = app_type
            return self

        def app_ticket(self, app_ticket: str) -> "ClientBuilder":
            self._config.app_ticket = app_ticket
            return self

        def enable_set_token(self, enable_set_token: bool) -> "ClientBuilder":
            self._config.enable_set_token = enable_set_token
            return self

        def cache(self, cache: ICache) -> "ClientBuilder":
            self._config.cache = cache
            return self

        def log_level(self, log_level: LogLevel) -> "ClientBuilder":
            self._config.log_level = log_level
            return self

        def source(self, source: str) -> "ClientBuilder":
            self._config.source = source
            return self

        def _init_cache(self) -> None:
            if self._config.cache is not None:
                TokenManager.cache = self._config.cache

        def _init_logger(self) -> None:
            logger.setLevel(int(self._config.log_level.value))

        def build(self) -> Client:
            client = Client()
            client._config = self._config
            self._init_cache()
            self._init_logger()
            client._lazy_services_enabled = True
            if AccessToken is not None:
                client.access_token = AccessToken(self._config)
            return client

    mod = types.ModuleType(f"{pkg_prefix}.client")
    mod.__file__ = str(pkg_root / "client.py")
    mod.__package__ = pkg_prefix
    mod.Client = Client
    mod.ClientBuilder = ClientBuilder
    mod._validate_domain = _validate_domain
    mod.Config = Config
    mod.BaseRequest = BaseRequest
    mod.BaseResponse = BaseResponse
    mod.RequestOption = RequestOption
    mod.TokenManager = TokenManager
    mod.Transport = _core_http.Transport
    mod.verify = _core_token.verify
    mod.JSON = JSON
    mod.logger = logger
    mod.AppType = AppType
    mod.LogLevel = LogLevel
    mod.Files = Files
    mod.ICache = ICache
    if AccessToken is not None:
        mod.AccessToken = AccessToken
    return mod


def _build_lazy_decryptor_module(pkg_root: Path, *, pkg_prefix: str = "lark_oapi") -> types.ModuleType:
    """Construct a lazy ``<pkg_prefix>.core.utils.decryptor`` deferring ``Crypto``."""
    const_mod = importlib.import_module(f"{pkg_prefix}.core.const")
    UTF_8 = const_mod.UTF_8

    class AESCipher(object):
        def __init__(self, key: Any) -> None:
            import hashlib

            if isinstance(key, str):
                key = key.encode(UTF_8)
            self.digest = hashlib.sha256(key).digest()

        def decrypt(self, enc: bytes) -> bytes:
            from Crypto.Cipher import AES

            if len(enc) < AES.block_size * 2 or len(enc) % AES.block_size != 0:
                raise ValueError("invalid ciphertext length")
            iv = enc[: AES.block_size]
            cipher = AES.new(self.digest, AES.MODE_CBC, iv)
            s = cipher.decrypt(enc[AES.block_size :])
            if not s:
                raise ValueError("invalid ciphertext: empty plaintext")
            pad = s[-1]
            if pad < 1 or pad > AES.block_size or pad > len(s):
                raise ValueError("invalid PKCS7 padding")
            if s[-pad:] != bytes([pad]) * pad:
                raise ValueError("invalid PKCS7 padding")
            return s[:-pad]

        def decrypt_str(self, enc: str) -> str:
            import base64

            raw = base64.b64decode(enc)
            return self.decrypt(raw).decode(UTF_8)

    mod = types.ModuleType(f"{pkg_prefix}.core.utils.decryptor")
    mod.__file__ = str(pkg_root / "core" / "utils" / "decryptor.py")
    mod.__package__ = f"{pkg_prefix}.core.utils"
    mod.AESCipher = AESCipher
    mod.__all__ = ["AESCipher"]
    return mod


def _httpx_client_kwargs_for(conf: Any, client_cls: Any) -> dict[str, Any]:
    proxy_url = getattr(conf, "proxy_url", None)
    trust_env = getattr(conf, "trust_env_proxy", None)
    kwargs: dict[str, Any] = {}
    if proxy_url:
        params = inspect.signature(client_cls).parameters
        if "proxy" in params:
            kwargs["proxy"] = proxy_url
        elif "proxies" in params:
            kwargs["proxies"] = proxy_url
        else:
            raise RuntimeError("installed httpx does not support explicit proxy configuration")
    if trust_env is not None:
        kwargs["trust_env"] = trust_env
    return kwargs


def _build_lazy_transport_module(pkg_root: Path, *, pkg_prefix: str = "lark_oapi") -> types.ModuleType:
    """Construct an ``httpx``-unified ``<pkg_prefix>.core.http.transport`` module."""
    const_mod = importlib.import_module(f"{pkg_prefix}.core.const")
    UTF_8 = const_mod.UTF_8
    USER_AGENT = const_mod.USER_AGENT
    AUTHORIZATION = const_mod.AUTHORIZATION
    enum_mod = importlib.import_module(f"{pkg_prefix}.core.enum")
    AccessTokenType = enum_mod.AccessTokenType
    JSON = importlib.import_module(f"{pkg_prefix}.core.json").JSON
    log_mod = importlib.import_module(f"{pkg_prefix}.core.log")
    sdk_logger = log_mod.logger
    redact_for_log = getattr(log_mod, "redact_for_log", None)
    redact_files_for_log = getattr(log_mod, "redact_files_for_log", None)
    redact_query_params_for_log = getattr(log_mod, "redact_query_params_for_log", None)
    model_mod = importlib.import_module(f"{pkg_prefix}.core.model")
    Config = model_mod.Config
    BaseRequest = model_mod.BaseRequest
    RawResponse = model_mod.RawResponse
    RequestOption = model_mod.RequestOption
    build_user_agent = importlib.import_module(
        f"{pkg_prefix}.core.utils.user_agent"
    ).build_user_agent

    def _build_url(domain: str, uri: str, paths: dict[str, str]) -> str:
        if paths is None:
            paths = {}
        for key in paths:
            value = paths[key]
            if value is None:
                value = ""
            encoded = urllib.parse.quote(str(value), safe="")
            uri = uri.replace(":" + key, encoded)
        if uri.startswith("http://") or uri.startswith("https://"):
            return uri
        return domain + uri

    def _build_header(
        request: Any, option: Any, conf: Optional[Any] = None
    ) -> dict[str, str]:
        headers = request.headers
        source = getattr(conf, "source", None) if conf is not None else None
        extra_tags = getattr(conf, "extra_ua_tags", None) if conf is not None else None
        headers[USER_AGENT] = build_user_agent(source=source, extra_tags=extra_tags)
        if option.headers is not None:
            for key in option.headers:
                headers[key] = option.headers[key]
        for token_type in request.token_types:
            if AccessTokenType.TENANT == token_type:
                headers[AUTHORIZATION] = f"Bearer {option.tenant_access_token}"
            elif AccessTokenType.APP == token_type:
                headers[AUTHORIZATION] = f"Bearer {option.app_access_token}"
            elif AccessTokenType.USER == token_type:
                headers[AUTHORIZATION] = f"Bearer {option.user_access_token}"
        return headers

    def _merge_dicts(*dicts: Any) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for d in dicts:
            if d is not None:
                res.update(d)
        return res

    def _redact_body_for_log(json_: Any, files: Any, data: Any) -> dict[str, Any]:
        if callable(redact_for_log) and callable(redact_files_for_log):
            return _merge_dicts(
                redact_for_log(json_),
                redact_files_for_log(files),
                redact_for_log(data),
            )
        return {}

    class Transport(object):
        @staticmethod
        def execute(
            conf: Any, req: Any, option: Optional[Any] = None
        ) -> Any:
            import httpx

            if option is None:
                option = RequestOption()
            url = _build_url(conf.domain, req.uri, req.paths)
            headers = _build_header(req, option, conf)

            data = req.body
            content: Optional[bytes] = None
            if data is not None:
                to_string = getattr(data, "to_string", None)
                if callable(to_string):
                    content = to_string()
                elif isinstance(data, (bytes, bytearray)):
                    content = bytes(data)
                else:
                    content = JSON.marshal(req.body).encode(UTF_8)

            client_kwargs = _httpx_client_kwargs_for(conf, httpx.Client)
            with httpx.Client(**client_kwargs) as client:
                response = client.request(
                    str(req.http_method.name),
                    url,
                    headers=req.headers,
                    params=req.queries,
                    content=content,
                    timeout=conf.timeout,
                )

            if sdk_logger.isEnabledFor(logging.DEBUG):
                if callable(redact_for_log) and callable(redact_query_params_for_log):
                    sdk_logger.debug(
                        f"{str(req.http_method.name)} {url} {response.status_code}, "
                        f"headers: {JSON.marshal(redact_for_log(headers))}, "
                        f"params: {JSON.marshal(redact_query_params_for_log(req.queries))}, "
                        f"body: {JSON.marshal(redact_for_log(content))}"
                    )
                else:
                    sdk_logger.debug(
                        f"{str(req.http_method.name)} request completed with status {response.status_code}, "
                        f"headers_count: {len(headers)}, "
                        f"params_count: {len(req.queries)}, "
                        f"body_present: {content is not None}"
                    )

            resp = RawResponse()
            resp.status_code = response.status_code
            resp.headers = dict(response.headers)
            resp.content = response.content
            return resp

        @staticmethod
        async def aexecute(
            conf: Any, req: Any, option: Optional[Any] = None
        ) -> Any:
            import httpx

            if option is None:
                option = RequestOption()
            url = _build_url(conf.domain, req.uri, req.paths)
            headers = _build_header(req, option, conf)

            json_, files, data = None, None, None
            if req.files:
                files = req.files
                if req.body is not None:
                    data = json.loads(JSON.marshal(req.body))
            elif req.body is not None:
                json_ = json.loads(JSON.marshal(req.body))

            client_kwargs = _httpx_client_kwargs_for(conf, httpx.AsyncClient)
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.request(
                    str(req.http_method.name),
                    url,
                    headers=req.headers,
                    params=req.queries,
                    json=json_,
                    data=data,
                    files=files,
                    timeout=conf.timeout,
                )

                if sdk_logger.isEnabledFor(logging.DEBUG):
                    if callable(redact_for_log) and callable(redact_query_params_for_log):
                        sdk_logger.debug(
                            f"{str(req.http_method.name)} {url} {response.status_code}"
                            f"{f', headers: {JSON.marshal(redact_for_log(headers))}' if headers else ''}"
                            f"{f', params: {JSON.marshal(redact_query_params_for_log(req.queries))}' if req.queries else ''}"
                            f"{f', body: {JSON.marshal(_redact_body_for_log(json_, files, data))}' if json_ or files or data else ''}"
                        )
                    else:
                        sdk_logger.debug(
                            f"{str(req.http_method.name)} request completed with status {response.status_code}, "
                            f"headers_count: {len(headers)}, "
                            f"params_count: {len(req.queries)}, "
                            f"body_present: {json_ is not None or files is not None or data is not None}"
                        )

                resp = RawResponse()
                resp.status_code = response.status_code
                resp.headers = dict(response.headers)
                resp.content = response.content
                return resp

    mod = types.ModuleType(f"{pkg_prefix}.core.http.transport")
    mod.__file__ = str(pkg_root / "core" / "http" / "transport.py")
    mod.__package__ = f"{pkg_prefix}.core.http"
    mod.Transport = Transport
    mod.Config = Config
    mod.BaseRequest = BaseRequest
    mod.RawResponse = RawResponse
    mod.RequestOption = RequestOption
    mod._build_url = _build_url
    mod._build_header = _build_header
    mod._merge_dicts = _merge_dicts
    mod._redact_body_for_log = _redact_body_for_log
    return mod


class _LazyHttpPackage(_LazyLarkPackage):
    """Lazy package for ``<pkg_prefix>.core.http`` deferring HTTP stack imports."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name not in (
            "_build_url",
            "_build_header",
            "_merge_dicts",
            "_redact_body_for_log",
        ):
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

        with self._lazy_lock:
            if name in self.__dict__:
                return self.__dict__[name]

            if name in ("handler", "transport"):
                mod = importlib.import_module(f"{self.__name__}.{name}")
                self.__dict__[name] = mod
                return mod

            if name == "HttpHandler":
                handler_mod = importlib.import_module(f"{self.__name__}.handler")
                val = getattr(handler_mod, name)
                self.__dict__[name] = val
                return val

            if name in (
                "Transport",
                "_build_url",
                "_build_header",
                "_merge_dicts",
                "_redact_body_for_log",
            ):
                transport_mod = importlib.import_module(f"{self.__name__}.transport")
                val = getattr(transport_mod, name)
                self.__dict__[name] = val
                return val

            _exec_real_init(self, self._lazy_pkg_dir / "__init__.py")
            if name in self.__dict__:
                return self.__dict__[name]

        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


class _HttpxRequestsCompat:
    """Minimal ``requests``-compatible facade backed by lazy ``httpx`` for WS endpoint POST."""

    @staticmethod
    def _extract_proxy_url(proxies: Any) -> Optional[str]:
        if isinstance(proxies, str):
            return proxies
        if isinstance(proxies, dict):
            return proxies.get("https") or proxies.get("http")
        return None

    class Session:
        def __init__(self) -> None:
            self.trust_env: bool = True

        def post(
            self,
            url: str,
            *,
            headers: Any = None,
            json: Any = None,
            proxies: Any = None,
            timeout: Any = None,
            **kwargs: Any,
        ) -> Any:
            import httpx

            client_kwargs: dict[str, Any] = {"trust_env": self.trust_env}
            proxy_url = _HttpxRequestsCompat._extract_proxy_url(proxies)
            if proxy_url:
                params = inspect.signature(httpx.Client).parameters
                if "proxy" in params:
                    client_kwargs["proxy"] = proxy_url
                elif "proxies" in params:
                    client_kwargs["proxies"] = proxy_url
                else:
                    raise RuntimeError(
                        "installed httpx does not support explicit proxy configuration"
                    )
            with httpx.Client(**client_kwargs) as client:
                return client.post(
                    url,
                    headers=headers,
                    json=json,
                    timeout=timeout,
                    **kwargs,
                )

    @classmethod
    def post(
        cls,
        url: str,
        *,
        headers: Any = None,
        json: Any = None,
        proxies: Any = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> Any:
        return cls.Session().post(
            url,
            headers=headers,
            json=json,
            proxies=proxies,
            timeout=timeout,
            **kwargs,
        )


class _LazyChannelWSClientLoader(importlib.abc.Loader):
    """Execute ``lark_channel/ws/client.py`` without loading duplicate ``requests`` stack."""

    def __init__(self, client_file: Path) -> None:
        self._client_file = client_file

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        source = self._client_file.read_text(encoding="utf-8")
        if "requests" not in sys.modules:
            try:
                tree = ast.parse(source, str(self._client_file))
                replaced = False
                new_body: list[ast.stmt] = []
                for stmt in tree.body:
                    if (
                        not replaced
                        and isinstance(stmt, ast.Import)
                        and len(stmt.names) == 1
                        and stmt.names[0].name == "requests"
                        and stmt.names[0].asname is None
                    ):
                        assign = ast.Assign(
                            targets=[ast.Name(id="requests", ctx=ast.Store())],
                            value=ast.Name(id="_ghostap_requests", ctx=ast.Load()),
                        )
                        ast.copy_location(assign, stmt)
                        ast.fix_missing_locations(assign)
                        new_body.append(assign)
                        replaced = True
                    else:
                        new_body.append(stmt)
                if replaced:
                    tree.body = new_body
                    module.__dict__["_ghostap_requests"] = _HttpxRequestsCompat
                    code = compile(tree, str(self._client_file), "exec")
                    exec(code, module.__dict__)
                    return
            except Exception:
                pass
        code = compile(source, str(self._client_file), "exec")
        exec(code, module.__dict__)


class _BlockedSubmoduleLoader(importlib.abc.Loader):
    """Raise ``ImportError`` for unused CLI helper submodules like ``httpx._main``."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        raise ImportError(f"{module.__name__} CLI helper disabled in GhostAP runtime")


class _BlockedSubmoduleFinder(importlib.abc.MetaPathFinder):
    """Block optional CLI submodules that pull heavy formatting dependencies (click/pygments)."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        del path, target
        if fullname not in _BLOCKED_HTTPX_SUBMODULES:
            return None
        main_mod = sys.modules.get("__main__")
        if getattr(main_mod, "__package__", None) == "httpx":
            return None
        return importlib.util.spec_from_loader(
            fullname, _BlockedSubmoduleLoader(), is_package=False
        )


class _LazyLarkLoader(importlib.abc.Loader):
    """Loader creating lazy ``lark_oapi`` / ``lark_channel`` package or ``client`` modules."""

    def __init__(
        self,
        pkg_root: Path,
        pkg_dir: Optional[Path] = None,
        *,
        pkg_prefix: str = "lark_oapi",
    ) -> None:
        self._pkg_root = pkg_root
        self._pkg_dir = pkg_dir
        self._pkg_prefix = pkg_prefix

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        fullname = spec.name
        if fullname == self._pkg_prefix:
            if self._pkg_prefix == "lark_channel":
                return _LazyChannelRootModule(fullname, self._pkg_root)
            return _LazyLarkRootModule(fullname, self._pkg_root)
        if fullname == f"{self._pkg_prefix}.client":
            return _build_lazy_client_module(self._pkg_root, pkg_prefix=self._pkg_prefix)
        if fullname == f"{self._pkg_prefix}.core.utils.decryptor":
            return _build_lazy_decryptor_module(self._pkg_root, pkg_prefix=self._pkg_prefix)
        if fullname == f"{self._pkg_prefix}.core.http.transport":
            return _build_lazy_transport_module(self._pkg_root, pkg_prefix=self._pkg_prefix)
        parts = fullname.split(".")
        assert self._pkg_dir is not None
        if fullname == f"{self._pkg_prefix}.core.http":
            return _LazyHttpPackage(fullname, self._pkg_dir)
        if self._pkg_prefix == "lark_channel" and len(parts) == 2 and parts[1] == "channel":
            return _LazyChannelSubpackage(fullname, self._pkg_dir)
        if len(parts) == 2 and parts[1] == "api":
            return _LazySubmodulePackage(fullname, self._pkg_dir)
        if len(parts) == 3 and parts[1] == "api":
            return _LazySubmodulePackage(fullname, self._pkg_dir)
        if len(parts) == 4 and parts[1] == "api":
            return _LazyVersionPackage(fullname, self._pkg_dir)
        if len(parts) == 5 and parts[1] == "api" and parts[4] == "model":
            return _LazyModelPackage(fullname, self._pkg_dir)
        if len(parts) == 5 and parts[1] == "api" and parts[4] == "resource":
            return _LazyModelPackage(fullname, self._pkg_dir)
        raise ImportError(f"Unsupported lazy lark module: {fullname}")

    def exec_module(self, module: types.ModuleType) -> None:
        return None


class _HydratingVersionLoader(importlib.abc.Loader):
    """Execute ``resource/__init__.py`` before real ``version.py`` when imported directly."""

    def __init__(self, res_pkg_name: str, res_init: Path, version_file: Path) -> None:
        self._res_pkg_name = res_pkg_name
        self._res_init = res_init
        self._version_file = version_file

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        res_pkg = importlib.import_module(self._res_pkg_name)
        _exec_real_init(res_pkg, self._res_init)
        code = compile(self._version_file.read_text(encoding="utf-8"), str(self._version_file), "exec")
        exec(code, module.__dict__)


class _LazyLarkFinder(importlib.abc.MetaPathFinder):
    """``sys.meta_path`` finder intercepting eager ``lark_oapi`` or ``lark_channel`` packages."""

    def __init__(self, pkg_root: Path, *, pkg_prefix: str = "lark_oapi") -> None:
        self._pkg_root = pkg_root
        self._pkg_prefix = pkg_prefix

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        del path, target
        if fullname != self._pkg_prefix and not fullname.startswith(f"{self._pkg_prefix}."):
            return None

        if fullname == self._pkg_prefix:
            loader = _LazyLarkLoader(
                self._pkg_root, self._pkg_root, pkg_prefix=self._pkg_prefix
            )
            return importlib.util.spec_from_loader(fullname, loader, is_package=True)

        if fullname == f"{self._pkg_prefix}.client":
            loader = _LazyLarkLoader(self._pkg_root, pkg_prefix=self._pkg_prefix)
            return importlib.util.spec_from_loader(fullname, loader, is_package=False)

        if fullname == f"{self._pkg_prefix}.core.utils.decryptor":
            decryptor_file = self._pkg_root / "core" / "utils" / "decryptor.py"
            if decryptor_file.is_file():
                loader = _LazyLarkLoader(self._pkg_root, pkg_prefix=self._pkg_prefix)
                return importlib.util.spec_from_loader(
                    fullname, loader, origin=str(decryptor_file), is_package=False
                )

        if fullname == f"{self._pkg_prefix}.core.http":
            http_dir = self._pkg_root / "core" / "http"
            if http_dir.is_dir():
                loader = _LazyLarkLoader(
                    self._pkg_root, http_dir, pkg_prefix=self._pkg_prefix
                )
                return importlib.util.spec_from_loader(fullname, loader, is_package=True)

        if fullname == f"{self._pkg_prefix}.core.http.transport":
            transport_file = self._pkg_root / "core" / "http" / "transport.py"
            if transport_file.is_file():
                loader = _LazyLarkLoader(self._pkg_root, pkg_prefix=self._pkg_prefix)
                return importlib.util.spec_from_loader(
                    fullname, loader, origin=str(transport_file), is_package=False
                )

        if fullname == "lark_channel.ws.client":
            ws_client_file = self._pkg_root / "ws" / "client.py"
            if ws_client_file.is_file():
                return importlib.util.spec_from_loader(
                    fullname,
                    _LazyChannelWSClientLoader(ws_client_file),
                    origin=str(ws_client_file),
                    is_package=False,
                )

        parts = fullname.split(".")
        if self._pkg_prefix == "lark_channel" and len(parts) == 2 and parts[1] == "channel":
            pkg_dir = self._pkg_root / "channel"
            if pkg_dir.is_dir():
                return importlib.util.spec_from_loader(
                    fullname,
                    _LazyLarkLoader(self._pkg_root, pkg_dir, pkg_prefix=self._pkg_prefix),
                    is_package=True,
                )

        if len(parts) >= 2 and parts[1] == "api":
            if len(parts) == 2:
                pkg_dir = self._pkg_root / "api"
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname,
                        _LazyLarkLoader(self._pkg_root, pkg_dir, pkg_prefix=self._pkg_prefix),
                        is_package=True,
                    )
            elif len(parts) == 3:
                pkg_dir = self._pkg_root / "api" / parts[2]
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname,
                        _LazyLarkLoader(self._pkg_root, pkg_dir, pkg_prefix=self._pkg_prefix),
                        is_package=True,
                    )
            elif len(parts) == 4 and parts[3] != "service":
                pkg_dir = self._pkg_root / "api" / parts[2] / parts[3]
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname,
                        _LazyLarkLoader(self._pkg_root, pkg_dir, pkg_prefix=self._pkg_prefix),
                        is_package=True,
                    )
            elif len(parts) == 5 and parts[4] in ("model", "resource"):
                pkg_dir = self._pkg_root / "api" / parts[2] / parts[3] / parts[4]
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname,
                        _LazyLarkLoader(self._pkg_root, pkg_dir, pkg_prefix=self._pkg_prefix),
                        is_package=True,
                    )
            elif len(parts) == 5 and parts[4] == "version":
                ver_dir = self._pkg_root / "api" / parts[2] / parts[3]
                ver_file = ver_dir / "version.py"
                res_init = ver_dir / "resource" / "__init__.py"
                if ver_file.is_file() and res_init.is_file():
                    res_pkg_name = f"{self._pkg_prefix}.api.{parts[2]}.{parts[3]}.resource"
                    return importlib.util.spec_from_loader(
                        fullname,
                        _HydratingVersionLoader(res_pkg_name, res_init, ver_file),
                        origin=str(ver_file),
                        is_package=False,
                    )

        return None


def _install_lazy_sdk(pkg_prefix: str, root_marker_attr: str) -> bool:
    with _INSTALL_LOCK:
        existing = sys.modules.get(pkg_prefix)
        if existing is not None:
            return bool(getattr(existing, root_marker_attr, False))
        if any(name.startswith(f"{pkg_prefix}.") for name in sys.modules):
            return False

        if any(
            isinstance(f, _LazyLarkFinder) and f._pkg_prefix == pkg_prefix
            for f in sys.meta_path
        ):
            return True

        spec = importlib.machinery.PathFinder.find_spec(pkg_prefix)
        if spec is None or not spec.submodule_search_locations:
            return False

        pkg_root = Path(next(iter(spec.submodule_search_locations)))
        if not pkg_root.is_dir():
            return False

        sys.meta_path.insert(0, _LazyLarkFinder(pkg_root, pkg_prefix=pkg_prefix))
        return True


def install_lazy_lark_oapi() -> bool:
    """Install lazy loading for ``lark_oapi`` if it has not been imported yet.

    Returns ``True`` when the lazy loader is active (newly installed or already
    installed as lazy root), or ``False`` if real ``lark_oapi`` was imported
    earlier or the SDK is unavailable.
    """
    return _install_lazy_sdk("lark_oapi", "_IS_LAZY_LARK_ROOT")


def install_lazy_lark_channel() -> bool:
    """Install lazy loading for ``lark_channel`` if it has not been imported yet."""
    return _install_lazy_sdk("lark_channel", "_IS_LAZY_LARK_CHANNEL_ROOT")


def install_lazy_httpx_guard() -> bool:
    """Prevent ``import httpx`` from eagerly loading optional CLI deps (click/pygments/rich)."""
    with _INSTALL_LOCK:
        if "httpx._main" in sys.modules:
            return False
        if any(isinstance(f, _BlockedSubmoduleFinder) for f in sys.meta_path):
            return True
        sys.meta_path.insert(0, _BlockedSubmoduleFinder())
        return True


__all__ = [
    "install_lazy_httpx_guard",
    "install_lazy_lark_channel",
    "install_lazy_lark_oapi",
]
