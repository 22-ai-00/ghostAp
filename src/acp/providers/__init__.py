from __future__ import annotations

import functools
import json
import logging
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..provider import ACPProvider, ToolRegistry, tool_registry

logger = logging.getLogger(__name__)

CODEX_ACP_NPM_PACKAGE = "@agentclientprotocol/codex-acp@1.1.7"


def _detect_model_arg_style(help_blob: str) -> str:
    blob = (help_blob or "").lower()
    if not blob:
        return "unknown"
    if "--model" in blob:
        return "model_long"
    if re.search(r"(^|\s)-m(\s|$)", blob):
        return "model_short"
    if "model.name" in blob or re.search(r"(^|\s)-c(\s|$)", blob):
        return "config_c"
    return "unknown"


AvailabilityChecker = Callable[[], bool]
HelpBlobLoader = Callable[[], str]


def _find_lru_cached(*fns: object) -> Optional[Callable[[], None]]:
    """Walk *fns* looking for the first ``lru_cache``-wrapped callable and
    return its ``cache_clear`` handle.  Inspects both the function itself and
    the cells of its ``__closure__`` (one level deep) so that it works for
    both raw ``@lru_cache`` functions and closures that capture one."""
    for fn in fns:
        if hasattr(fn, "cache_clear"):
            return fn.cache_clear  # type: ignore[union-attr]
        for cell in (getattr(fn, "__closure__", None) or []):
            obj = cell.cell_contents
            if callable(obj) and hasattr(obj, "cache_clear"):
                return obj.cache_clear
    return None


def _make_resolve_checker(tool_name: str) -> AvailabilityChecker:
    def _check() -> bool:
        from ..sync_adapter import _resolve_with_auto_update
        return _resolve_with_auto_update(tool_name)
    return _check


def _make_custom_help_checker(
    help_cmd: list[str],
    required_keywords: Sequence[str],
) -> tuple[AvailabilityChecker, HelpBlobLoader]:
    @functools.lru_cache(maxsize=1)
    def _get_help_blob() -> str:
        try:
            from ...utils.env import build_clean_env
            env = build_clean_env()
            p = subprocess.run(
                help_cmd,
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
            )
            return ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
        except Exception:
            return ""

    def _load_help_blob() -> str:
        blob = _get_help_blob()
        if blob:
            return blob
        _get_help_blob.cache_clear()
        return _get_help_blob()

    def _check() -> bool:
        blob = _load_help_blob().lower()
        return bool(blob and all(kw in blob for kw in required_keywords))

    return _check, _load_help_blob


def _make_probe_checker(tool_name: str) -> tuple[AvailabilityChecker, HelpBlobLoader]:
    @functools.lru_cache(maxsize=1)
    def _get_help_blob() -> str:
        try:
            from ..sync_adapter import _probe_acp_serve_help
            ok, _rc, out_snip, err_snip = _probe_acp_serve_help(tool_name)
            return (out_snip or "") + "\n" + (err_snip or "")
        except Exception:
            return ""

    def _check() -> bool:
        try:
            from ..sync_adapter import _probe_acp_serve_help
            ok, _rc, _out, _err = _probe_acp_serve_help(tool_name)
            return bool(ok)
        except Exception:
            return False

    return _check, _get_help_blob


@dataclass(frozen=True)
class _ProviderConfig:
    tool_name: str
    serve_args: list[str]
    availability_checker: AvailabilityChecker
    model_style: Optional[str] = None
    help_blob_loader: Optional[HelpBlobLoader] = None
    is_default: bool = False
    skip_model_selection: bool = False


def _apply_model_args(
    args: list[str],
    model_name: Optional[str],
    model_style: Optional[str],
    help_blob_loader: Optional[HelpBlobLoader],
) -> list[str]:
    m = (model_name or "").strip()
    if not m:
        return args

    if model_style is None:
        return args

    style = model_style
    if style == "dynamic" and help_blob_loader:
        style = _detect_model_arg_style(help_blob_loader())

    if style == "config_c":
        return args + ["-c", f"model.name={m}"]
    if style == "config_model":
        return args + ["-c", f"model={json.dumps(m)}"]
    if style == "model_long":
        return args + ["--model", m]
    if style == "model_short":
        return args + ["-m", m]
    return args


class GenericACPProvider:
    def __init__(self, config: _ProviderConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.tool_name

    @property
    def skip_model_selection(self) -> bool:
        return bool(self._config.skip_model_selection)

    def check_availability(self) -> bool:
        return self._config.availability_checker()

    def get_serve_command(self, model_name: Optional[str] = None) -> tuple[str, list[str]]:
        args = list(self._config.serve_args)
        args = _apply_model_args(
            args, self.normalize_model_name(model_name), self._config.model_style, self._config.help_blob_loader
        )
        return self._config.tool_name, args

    def get_fallback_command(self, model_name: Optional[str] = None) -> Optional[tuple[str, list[str]]]:
        return None

    def normalize_model_name(self, model_name: Optional[str] = None) -> Optional[str]:
        """Return the backend-facing model name for this provider."""
        if model_name is None:
            return None
        return str(model_name or "").strip() or None


class TraexACPProvider(GenericACPProvider):
    """Traex ACP provider with config_name → slug resolution.

    Traex's ACP config_options returns model values as config_name (e.g.
    "c_o_new_thinking") but its CLI metadata lookup uses slug format (e.g.
    "Test-O-New-Thinking"). This provider translates before passing -c model=.
    """

    @functools.lru_cache(maxsize=1)  # noqa: B019
    def _load_slug_map(self) -> dict[str, str]:
        """Build config_name → slug map from ~/.trae/cli/models_cache.json."""
        try:
            from pathlib import Path

            cache_path = Path.home() / ".trae" / "cli" / "models_cache.json"
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            mapping: dict[str, str] = {}
            for m in data.get("models") or []:
                slug = str(m.get("slug") or "").strip()
                config_name = str(m.get("config_name") or "").strip()
                if slug and config_name and config_name != slug:
                    mapping[config_name] = slug
            return mapping
        except Exception:
            return {}

    @classmethod
    def _strip_variant_suffix(cls, model_name: str) -> str:
        """Reduce "config_name/<profile>/<effort>" to the base config_name.

        Reuse the same parser as runtime application so startup normalization
        cannot drift from ACP session selection. Ordinary slash-bearing names
        (for example ``anthropic/claude-sonnet``) remain intact.
        """
        from ..traex_selection import split_traex_model_selection

        base, _profile, _effort = split_traex_model_selection(model_name)
        return base

    def normalize_model_name(self, model_name: Optional[str] = None) -> Optional[str]:
        """Resolve config_name to slug if a mapping exists.

        Compound cascade values ("config_name/profile/effort") are normalised to
        their base config_name first so the slug map (keyed by bare config_name)
        can match instead of passing an unknown compound name to the CLI.
        """
        m = (model_name or "").strip()
        if not m:
            return None
        base = self._strip_variant_suffix(m)
        slug_map = self._load_slug_map()
        if base in slug_map:
            return slug_map[base]
        # Fall back to the stripped base (never the compound) so the CLI receives
        # a name it can look up, even when no explicit slug mapping exists.
        return base or m

    def get_default_model(self) -> Optional[str]:
        """Resolve the first available model slug from cache when none specified.

        Intended for callers (e.g. create_engine_session) that need to supply a
        concrete model when the user did not select one. Not used during probe
        which deliberately passes model_name=None to get a bare serve command.
        """
        try:
            from pathlib import Path

            cache_path = Path.home() / ".trae" / "cli" / "models_cache.json"
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            for m in data.get("models") or []:
                slug = str(m.get("slug") or "").strip()
                if slug:
                    return slug
        except Exception:
            pass
        return None

    def get_serve_command(self, model_name: Optional[str] = None) -> tuple[str, list[str]]:
        resolved = self.normalize_model_name(model_name)
        args = list(self._config.serve_args)
        args = _apply_model_args(
            args, resolved, self._config.model_style, self._config.help_blob_loader
        )
        return self._config.tool_name, args


class CodexACPProvider(GenericACPProvider):
    """Codex ACP provider with an npx fallback for CLIs without `acp serve`."""

    def __init__(
        self,
        config: _ProviderConfig,
        *,
        fallback_package: str = CODEX_ACP_NPM_PACKAGE,
    ) -> None:
        super().__init__(config)
        self._fallback_package = fallback_package

    def _native_available(self) -> bool:
        try:
            return bool(self._config.availability_checker())
        except Exception:
            return False

    def _fallback_available(self) -> bool:
        return shutil.which("npx") is not None

    def check_availability(self) -> bool:
        return self._native_available() or self._fallback_available()

    def get_serve_command(self, model_name: Optional[str] = None) -> tuple[str, list[str]]:
        if self._native_available():
            return super().get_serve_command(model_name)
        fallback = self.get_fallback_command(model_name)
        if fallback:
            return fallback
        raise RuntimeError(
            "Codex ACP is unavailable: `codex acp serve` is not supported and `npx` was not found."
        )

    def get_fallback_command(self, model_name: Optional[str] = None) -> Optional[tuple[str, list[str]]]:
        if not self._fallback_available():
            return None
        # The official adapter is an ACP stdio server, not a Codex CLI wrapper.
        # It ignores Zed's historical ``-c model=...`` argv convention; the
        # selected model is applied through session/set_config_option after
        # new_session by SyncACPSession.
        return "npx", ["--yes", self._fallback_package]


def _make_custom_help_checker_with_cache_handle(
    help_cmd: list[str],
    required_keywords: Sequence[str],
) -> tuple[AvailabilityChecker, HelpBlobLoader, Callable[[], None]]:
    checker, loader = _make_custom_help_checker(help_cmd, required_keywords)
    clear_fn = _find_lru_cached(checker, loader)

    def _clear() -> None:
        if clear_fn is not None:
            clear_fn()

    return checker, loader, _clear


def _make_probe_checker_with_cache_handle(
    tool_name: str,
) -> tuple[AvailabilityChecker, HelpBlobLoader, Callable[[], None]]:
    checker, loader = _make_probe_checker(tool_name)
    clear_fn = _find_lru_cached(loader, checker)

    def _clear() -> None:
        if clear_fn is not None:
            clear_fn()

    return checker, loader, _clear


_checkers: dict[str, tuple] | None = None
_providers: dict[str, GenericACPProvider] | None = None
_init_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def _ensure_providers() -> dict[str, GenericACPProvider]:
    """Single lazy-init entry point: checkers → configs → providers."""
    global _checkers, _providers
    if _providers is not None:
        return _providers
    with _init_lock:
        if _providers is not None:
            return _providers

        # --- 1) build checkers ---
        aiden = _make_custom_help_checker_with_cache_handle(
            ["aiden", "acp", "--help"],
            ["usage:", "aiden acp", "acp agent"],
        )
        codex = _make_probe_checker_with_cache_handle("codex")
        gemini = _make_custom_help_checker_with_cache_handle(
            ["gemini", "--help"],
            ["usage:", "--acp"],
        )
        traex = _make_custom_help_checker_with_cache_handle(
            ["traex", "acp", "serve", "--help"],
            ["usage:", "acp serve"],
        )
        _checkers = {"aiden": aiden, "codex": codex, "gemini": gemini, "traex": traex}

        _aiden_checker, _aiden_help_loader, _ = aiden
        _codex_checker, _codex_help_loader, _ = codex
        _gemini_checker, _gemini_help_loader, _ = gemini
        _traex_checker, _traex_help_loader, _ = traex

        # --- 2) build configs ---
        configs = [
            _ProviderConfig(
                tool_name="coco",
                serve_args=["acp", "serve"],
                availability_checker=_make_resolve_checker("coco"),
                model_style="config_c",
                is_default=True,
                skip_model_selection=True,
            ),
            _ProviderConfig(
                tool_name="claude",
                serve_args=["acp", "serve"],
                availability_checker=_make_resolve_checker("claude"),
                # Claude Code CLI honours `--model <id>` (and the `[1m]` suffix
                # for 1M-context beta). Previously model_style=None silently
                # dropped the user's selection on the floor.
                model_style="model_long",
            ),
            _ProviderConfig(
                tool_name="aiden",
                serve_args=["acp"],
                availability_checker=_aiden_checker,
                model_style="dynamic",
                help_blob_loader=_aiden_help_loader,
                skip_model_selection=True,
            ),
            _ProviderConfig(
                tool_name="codex",
                serve_args=["acp", "serve"],
                availability_checker=_codex_checker,
                model_style="dynamic",
                help_blob_loader=_codex_help_loader,
            ),
            _ProviderConfig(
                tool_name="gemini",
                serve_args=["--acp"],
                availability_checker=_gemini_checker,
                model_style="dynamic",
                help_blob_loader=_gemini_help_loader,
            ),
            _ProviderConfig(
                tool_name="traex",
                serve_args=["acp", "serve"],
                availability_checker=_traex_checker,
                model_style="config_model",
                help_blob_loader=_traex_help_loader,
            ),
        ]

        # --- 3) build and register providers ---
        result: dict[str, GenericACPProvider] = {}
        for cfg in configs:
            if cfg.tool_name == "codex":
                p = CodexACPProvider(cfg)
            elif cfg.tool_name == "traex":
                p = TraexACPProvider(cfg)
            else:
                p = GenericACPProvider(cfg)
            result[cfg.tool_name] = p
            tool_registry.register(p, is_default=cfg.is_default)
        _providers = result
    return _providers


def _get_checker(name: str) -> tuple:
    _ensure_providers()  # guarantees _checkers is populated
    assert _checkers is not None
    return _checkers[name]


def get_providers() -> dict[str, GenericACPProvider]:
    """Public accessor — lazily builds and registers all providers on first call."""
    return _ensure_providers()


def normalize_acp_model_name(tool_name: str, model_name: Optional[str]) -> Optional[str]:
    """Normalize a selected ACP model into the backend-facing model identifier.

    UI model pickers may carry richer values such as Traex cascade variants
    (``config_name/profile/effort``). Provider-specific launch and protocol
    calls should use this helper so the backend receives a model it can resolve.
    """
    if model_name is None:
        return None
    model = str(model_name or "").strip()
    if not model:
        return None
    try:
        provider = get_providers().get(str(tool_name or "").strip().lower())
        normalize = getattr(provider, "normalize_model_name", None) if provider else None
        if callable(normalize):
            return normalize(model)
    except Exception:
        logger.debug("normalize_acp_model_name failed for tool=%s model=%s", tool_name, model_name, exc_info=True)
    return model


def _reset_providers_for_testing() -> None:
    """Reset providers, checkers, and their lru_caches. **Test-only.**

    After calling this, the next ``get_providers()`` / ``_ensure_providers()``
    will rebuild everything from scratch.
    """
    global _checkers, _providers
    with _init_lock:
        # Clear lru_caches held by checkers before discarding references.
        if _checkers is not None:
            for _name, entry in _checkers.items():
                _, _, clear_fn = entry
                try:
                    clear_fn()
                except Exception:
                    logger.debug("failed to clear lru_cache", exc_info=True)
        # Unregister providers from the module-level tool_registry (lock-safe).
        if _providers is not None:
            tool_registry._reset_for_testing(list(_providers.keys()))
        _checkers = None
        _providers = None


CocoProvider = type("CocoProvider", (), {"__new__": lambda cls: _ensure_providers()["coco"]})
ClaudeProvider = type("ClaudeProvider", (), {"__new__": lambda cls: _ensure_providers()["claude"]})
AidenProvider = type("AidenProvider", (), {"__new__": lambda cls: _ensure_providers()["aiden"]})
CodexProvider = type("CodexProvider", (), {"__new__": lambda cls: _ensure_providers()["codex"]})
GeminiProvider = type("GeminiProvider", (), {"__new__": lambda cls: _ensure_providers()["gemini"]})
TraexProvider = type("TraexProvider", (), {"__new__": lambda cls: _ensure_providers()["traex"]})


def _get_aiden_acp_serve_help_blob() -> str:
    _, loader, _ = _get_checker("aiden")
    return loader()

_get_aiden_acp_serve_help_blob.cache_clear = lambda: _get_checker("aiden")[2]()  # type: ignore[attr-defined]


def _get_codex_acp_serve_help_blob() -> str:
    _, loader, _ = _get_checker("codex")
    return loader()

_get_codex_acp_serve_help_blob.cache_clear = lambda: _get_checker("codex")[2]()  # type: ignore[attr-defined]


def _get_gemini_acp_serve_help_blob() -> str:
    _, loader, _ = _get_checker("gemini")
    return loader()

_get_gemini_acp_serve_help_blob.cache_clear = lambda: _get_checker("gemini")[2]()  # type: ignore[attr-defined]


def _get_traex_acp_serve_help_blob() -> str:
    _, loader, _ = _get_checker("traex")
    return loader()

_get_traex_acp_serve_help_blob.cache_clear = lambda: _get_checker("traex")[2]()  # type: ignore[attr-defined]


__all__ = [
    "ACPProvider",
    "ToolRegistry",
    "tool_registry",
    "get_providers",
    "normalize_acp_model_name",
    "_reset_providers_for_testing",
    "CocoProvider",
    "ClaudeProvider",
    "AidenProvider",
    "CodexProvider",
    "GeminiProvider",
    "TraexProvider",
    "GenericACPProvider",
    "CodexACPProvider",
    "CODEX_ACP_NPM_PACKAGE",
    "_get_aiden_acp_serve_help_blob",
    "_get_codex_acp_serve_help_blob",
    "_get_gemini_acp_serve_help_blob",
    "_get_traex_acp_serve_help_blob",
]
