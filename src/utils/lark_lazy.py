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

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import re
import sys
import threading
import types
import urllib.parse
from pathlib import Path
from typing import Any, Optional

_INSTALL_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

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

            _exec_real_init(self, pkg_dir / "__init__.py")
            if name in self.__dict__:
                return self.__dict__[name]

        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


class _LazyVersionProxy:
    """Lazy proxy for ``client.<svc>.<ver>`` (loads only accessed resources)."""

    def __init__(self, config: Any, svc_name: str, ver_name: str, ver_dir: Path) -> None:
        self._config = config
        self._svc_name = svc_name
        self._ver_name = ver_name
        self._ver_dir = ver_dir
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
                mod_name = f"lark_oapi.api.{self._svc_name}.{self._ver_name}.resource.{name}"
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
                        f"lark_oapi.api.{self._svc_name}.{self._ver_name}.resource"
                    )
                    _exec_real_init(res_pkg, self._ver_dir / "resource" / "__init__.py")
                    ver_mod = importlib.import_module(
                        f"lark_oapi.api.{self._svc_name}.{self._ver_name}.version"
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

    def __init__(self, config: Any, svc_name: str, svc_dir: Path) -> None:
        self._config = config
        self._svc_name = svc_name
        self._svc_dir = svc_dir
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
                proxy = _LazyVersionProxy(self._config, self._svc_name, name, ver_dir)
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
                    svc_mod = importlib.import_module(f"lark_oapi.api.{self._svc_name}.service")
                    for obj in vars(svc_mod).values():
                        if isinstance(obj, type) and obj.__module__ == svc_mod.__name__:
                            self._fallback_service = obj(self._config)
                            break
                if self._fallback_service is not None and hasattr(self._fallback_service, name):
                    val = getattr(self._fallback_service, name)
                    self.__dict__[name] = val
                    return val

        raise AttributeError(f"service '{self._svc_name}' has no attribute {name!r}")


def _build_lazy_client_module(lark_root: Path) -> types.ModuleType:
    """Construct the lazy drop-in replacement for ``lark_oapi.client``."""
    from lark_oapi.core import JSON, logger
    from lark_oapi.core import http as _core_http
    from lark_oapi.core import token as _core_token
    from lark_oapi.core.access_token import AccessToken
    from lark_oapi.core.cache import ICache
    from lark_oapi.core.const import APPLICATION_JSON, CONTENT_TYPE, UTF_8
    from lark_oapi.core.enum import AppType, LogLevel
    from lark_oapi.core.model import BaseRequest, BaseResponse, Config, RequestOption
    from lark_oapi.core.token import TokenManager
    from lark_oapi.core.utils.files import Files

    api_root = lark_root / "api"

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
        """Drop-in lazy replacement for ``lark_oapi.Client``."""

        def __init__(self) -> None:
            self._config: Optional[Config] = None
            self.access_token: Optional[AccessToken] = None
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
                    proxy = _LazyServiceProxy(self._config, name, svc_dir)
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
        """Drop-in replacement for ``lark_oapi.client.ClientBuilder``."""

        def __init__(self) -> None:
            self._config = Config()

        def app_id(self, app_id: str) -> "ClientBuilder":
            self._config.app_id = app_id
            return self

        def app_secret(self, app_secret: str) -> "ClientBuilder":
            self._config.app_secret = app_secret
            return self

        def client_assertion_provider(self, provider: Any) -> "ClientBuilder":
            self._config.client_assertion_provider = provider
            return self

        def oauth_base_url(self, oauth_base_url: str) -> "ClientBuilder":
            self._config.oauth_base_url = oauth_base_url
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
            client.access_token = AccessToken(self._config)
            return client

    mod = types.ModuleType("lark_oapi.client")
    mod.__file__ = str(lark_root / "client.py")
    mod.__package__ = "lark_oapi"
    mod.Client = Client
    mod.ClientBuilder = ClientBuilder
    mod._validate_domain = _validate_domain
    return mod


class _LazyLarkLoader(importlib.abc.Loader):
    """Loader creating lazy ``lark_oapi`` package or ``client`` modules."""

    def __init__(self, lark_root: Path, pkg_dir: Optional[Path] = None) -> None:
        self._lark_root = lark_root
        self._pkg_dir = pkg_dir

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        fullname = spec.name
        if fullname == "lark_oapi":
            return _LazyLarkRootModule(fullname, self._lark_root)
        if fullname == "lark_oapi.client":
            return _build_lazy_client_module(self._lark_root)
        parts = fullname.split(".")
        assert self._pkg_dir is not None
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
    """``sys.meta_path`` finder intercepting eager ``lark_oapi`` packages."""

    def __init__(self, lark_root: Path) -> None:
        self._lark_root = lark_root

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        del path, target
        if fullname != "lark_oapi" and not fullname.startswith("lark_oapi."):
            return None

        if fullname == "lark_oapi":
            loader = _LazyLarkLoader(self._lark_root, self._lark_root)
            return importlib.util.spec_from_loader(fullname, loader, is_package=True)

        if fullname == "lark_oapi.client":
            loader = _LazyLarkLoader(self._lark_root)
            return importlib.util.spec_from_loader(fullname, loader, is_package=False)

        parts = fullname.split(".")
        if len(parts) >= 2 and parts[1] == "api":
            if len(parts) == 2:
                pkg_dir = self._lark_root / "api"
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname, _LazyLarkLoader(self._lark_root, pkg_dir), is_package=True
                    )
            elif len(parts) == 3:
                pkg_dir = self._lark_root / "api" / parts[2]
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname, _LazyLarkLoader(self._lark_root, pkg_dir), is_package=True
                    )
            elif len(parts) == 4 and parts[3] != "service":
                pkg_dir = self._lark_root / "api" / parts[2] / parts[3]
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname, _LazyLarkLoader(self._lark_root, pkg_dir), is_package=True
                    )
            elif len(parts) == 5 and parts[4] in ("model", "resource"):
                pkg_dir = self._lark_root / "api" / parts[2] / parts[3] / parts[4]
                if pkg_dir.is_dir():
                    return importlib.util.spec_from_loader(
                        fullname, _LazyLarkLoader(self._lark_root, pkg_dir), is_package=True
                    )
            elif len(parts) == 5 and parts[4] == "version":
                ver_dir = self._lark_root / "api" / parts[2] / parts[3]
                ver_file = ver_dir / "version.py"
                res_init = ver_dir / "resource" / "__init__.py"
                if ver_file.is_file() and res_init.is_file():
                    res_pkg_name = f"lark_oapi.api.{parts[2]}.{parts[3]}.resource"
                    return importlib.util.spec_from_loader(
                        fullname,
                        _HydratingVersionLoader(res_pkg_name, res_init, ver_file),
                        origin=str(ver_file),
                        is_package=False,
                    )

        return None


def install_lazy_lark_oapi() -> bool:
    """Install lazy loading for ``lark_oapi`` if it has not been imported yet.

    Returns ``True`` when the lazy loader is active (newly installed or already
    installed as lazy root), or ``False`` if real ``lark_oapi`` was imported
    earlier or the SDK is unavailable.
    """
    with _INSTALL_LOCK:
        existing = sys.modules.get("lark_oapi")
        if existing is not None:
            return bool(getattr(existing, "_IS_LAZY_LARK_ROOT", False))
        if any(name.startswith("lark_oapi.") for name in sys.modules):
            return False

        # Already installed finder, module not yet imported
        if any(isinstance(f, _LazyLarkFinder) for f in sys.meta_path):
            return True

        spec = importlib.machinery.PathFinder.find_spec("lark_oapi")
        if spec is None or not spec.submodule_search_locations:
            return False

        lark_root = Path(next(iter(spec.submodule_search_locations)))
        if not lark_root.is_dir():
            return False

        sys.meta_path.insert(0, _LazyLarkFinder(lark_root))
        return True


__all__ = ["install_lazy_lark_oapi"]
