"""GhostAP command-line entry point (``ghostap``)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from pathlib import Path
from typing import Optional

from .mode.backends import VISIBLE_BACKEND_SPECS
from .workflow_engine.constants import NODE_MIN_VERSION, RUNTIME_JS_PATH

_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "your_app_id",
        "your_app_secret",
        "changeme",
        "xxx",
    }
)

_BUILTIN_ENV_TEMPLATE = """\
# GhostAP 快速开始配置文件
# 请填入飞书开放平台应用的 APP_ID 与 APP_SECRET，权限建议保持 0600

# ── 飞书应用配置（必填）──
APP_ID=your_app_id
APP_SECRET=your_app_secret

# ── 默认 ACP 工具（coco/claude/aiden/codex/gemini/traex/grok/dsh/agy，留空为纯 Shell）──
DEFAULT_ACP_TOOL=coco

# ── 管理员（在飞书私聊 Bot 发送 /setadmin 自动绑定）──
ADMIN_USER_IDS=

# ── 飞书入口访问 ──
INGRESS_ACCESS_MODE=legacy_allow_all
ADMIN_BOOTSTRAP_SCOPE=p2p_only

# ── 消息回复与超时 ──
DEFAULT_REPLY_MODE=thread
COMMAND_TIMEOUT=30
COCO_EXECUTION_TIMEOUT=7200
WORKFLOW_TOTAL_TIMEOUT_S=3600
WORKFLOW_AGENT_CALL_TIMEOUT_S=600
"""


def _get_version() -> str:
    """Return installed package version, falling back to pyproject.toml or ``src.__version__``."""
    try:
        return metadata_version("ghostap")
    except PackageNotFoundError:
        pass

    try:
        import tomllib

        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject_path.is_file():
            with open(pyproject_path, "rb") as handle:
                data = tomllib.load(handle)
            pyproject_version = str(data.get("project", {}).get("version", "")).strip()
            if pyproject_version:
                return pyproject_version
    except Exception:
        pass

    from . import __version__

    return __version__


def _find_env_example(target_dir: Path) -> Optional[Path]:
    """Locate ``.env.example`` from target directory, CWD, or repository checkout."""
    candidates = [
        target_dir / ".env.example",
        Path.cwd() / ".env.example",
        Path(__file__).resolve().parent.parent / ".env.example",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _write_private_file(path: Path, content: str) -> None:
    """Write text file with ``0600`` permissions across Linux and macOS."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(path, 0o600)


def init_env(target_dir: Path, *, force: bool = False) -> int:
    """Initialize ``.env`` with ``0600`` permissions from ``.env.example`` or built-in template."""
    resolved_dir = target_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    env_path = resolved_dir / ".env"

    if env_path.exists() and not force:
        print(f"⚠️  .env 已存在（如需覆盖请使用 --force）: {env_path}")
        return 0

    template_path = _find_env_example(resolved_dir)
    if template_path is not None:
        content = template_path.read_text(encoding="utf-8")
        source_label = str(template_path)
    else:
        content = _BUILTIN_ENV_TEMPLATE
        source_label = "built-in template"

    _write_private_file(env_path, content)
    print(f"✅ 已生成配置文件: {env_path} (权限 0600, 来源: {source_label})")
    print("下一步:")
    print(f"  1. 编辑 {env_path} 并填入 APP_ID 和 APP_SECRET")
    print("  2. 运行 ghostap --validate 校验配置")
    print("  3. 运行 ghostap 启动服务，并在飞书私聊中发送 /setadmin 绑定管理员")
    return 0


def _parse_semver_tuple(raw: str) -> tuple[int, ...]:
    """Extract numeric ``(major, minor, patch)`` tuple from a version string."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", raw or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.groups() if part is not None)


def _probe_binary_version(cmd: str, args: Optional[list[str]] = None) -> tuple[bool, Optional[str], str]:
    """Check whether ``cmd`` is on PATH and capture its single-line version output."""
    binary = shutil.which(cmd)
    if not binary:
        return False, None, ""
    probe_args = args if args is not None else ["--version"]
    try:
        proc = subprocess.run(
            [binary, *probe_args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        raw_out = (proc.stdout or proc.stderr or "").strip()
        first_line = raw_out.splitlines()[0].strip() if raw_out else ""
        return True, binary, first_line
    except (OSError, subprocess.SubprocessError):
        return True, binary, ""


def _parse_env_file(env_path: Path) -> dict[str, Optional[str]]:
    """Parse key-value pairs from a ``.env`` file using ``python-dotenv`` or stdlib fallback."""
    try:
        from dotenv import dotenv_values

        return dict(dotenv_values(env_path))
    except ImportError:
        parsed: dict[str, Optional[str]] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            parsed[key] = value
        return parsed


def _inspect_env_status(target_dir: Path) -> tuple[bool, bool, bool, str]:
    """Check ``.env`` existence and whether ``APP_ID`` / ``APP_SECRET`` are configured."""
    env_path = target_dir / ".env"
    if not env_path.is_file():
        return False, False, False, f"missing ({env_path})"

    values = _parse_env_file(env_path)
    app_id = str(values.get("APP_ID") or "").strip()
    app_secret = str(values.get("APP_SECRET") or "").strip()

    app_id_ok = app_id.lower() not in _PLACEHOLDER_VALUES
    app_secret_ok = app_secret.lower() not in _PLACEHOLDER_VALUES

    if app_id_ok and app_secret_ok:
        return True, True, True, f"configured ({env_path})"
    missing_fields = []
    if not app_id_ok:
        missing_fields.append("APP_ID")
    if not app_secret_ok:
        missing_fields.append("APP_SECRET")
    return True, app_id_ok, app_secret_ok, f"incomplete: {', '.join(missing_fields)} ({env_path})"


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a clean ASCII/UTF-8 aligned diagnostic table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(values: tuple[str, ...]) -> str:
        return "  " + "  ".join(val.ljust(widths[idx]) for idx, val in enumerate(values)).rstrip()

    separator = "  " + "  ".join("-" * w for w in widths)
    lines = [_fmt_row(headers), separator, *(_fmt_row(r) for r in rows)]
    return "\n".join(lines)


def run_doctor(target_dir: Optional[Path] = None) -> int:
    """Print diagnostic report covering Python, uv, Node.js, ``.env``, and 9 AI CLIs."""
    resolved_dir = (target_dir or Path.cwd()).expanduser().resolve()
    ver = _get_version()
    print(f"GhostAP v{ver} Doctor")
    print("=" * 60)

    # 1. Core environment & prerequisites
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)

    uv_found, uv_path, uv_ver = _probe_binary_version("uv")
    node_found, node_path, node_ver = _probe_binary_version("node")
    node_tuple = _parse_semver_tuple(node_ver)
    node_ok = node_found and node_tuple >= NODE_MIN_VERSION
    min_node_str = ".".join(str(x) for x in NODE_MIN_VERSION)

    env_exists, app_id_ok, app_secret_ok, env_detail = _inspect_env_status(resolved_dir)
    env_ok = env_exists and app_id_ok and app_secret_ok
    runtime_ok = Path(RUNTIME_JS_PATH).is_file()

    env_rows: list[tuple[str, str, str]] = [
        (
            "Python (>= 3.11)",
            "OK" if py_ok else "FAIL",
            f"{py_ver} ({sys.executable})",
        ),
        (
            "uv",
            "OK" if uv_found else "WARN",
            f"{uv_ver or 'installed'} ({uv_path})" if uv_found else "not found in PATH",
        ),
        (
            f"Node.js (>= {min_node_str}, /wf)",
            "OK" if node_ok else ("WARN" if node_found else "MISSING"),
            (
                f"{node_ver or 'installed'} ({node_path})"
                if node_found
                else "not found (required only for /wf Teamwork)"
            ),
        ),
        (
            "Workflow Runtime JS",
            "OK" if runtime_ok else "FAIL",
            RUNTIME_JS_PATH if runtime_ok else f"missing ({RUNTIME_JS_PATH})",
        ),
        (
            ".env & Credentials",
            "OK" if env_ok else ("WARN" if env_exists else "MISSING"),
            env_detail,
        ),
    ]
    print("\n[Environment & Configuration]")
    print(_format_table(("Check", "Status", "Detail"), env_rows))

    # 2. 9 visible AI CLIs from SSOT backend registry
    cli_rows: list[tuple[str, str, str, str, str]] = []
    detected_count = 0
    for spec in VISIBLE_BACKEND_SPECS:
        bin_path = shutil.which(spec.cli_name)
        if bin_path:
            detected_count += 1
            status = "OK"
            location = bin_path
        else:
            status = "MISSING"
            location = "-"
        cli_rows.append((spec.display_name, spec.cli_name, spec.transport, status, location))

    print(f"\n[AI Coding Backends ({detected_count}/{len(VISIBLE_BACKEND_SPECS)} detected)]")
    print(_format_table(("Backend", "CLI", "Transport", "Status", "Location"), cli_rows))

    if not env_ok:
        print("\nHint: run `ghostap init` and set APP_ID / APP_SECRET in .env, then run `ghostap --validate`.")
    elif detected_count == 0:
        print("\nHint: install at least one AI CLI (e.g., coco, claude, codex, gemini, agy) to enable coding modes.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct top-level CLI argument parser for ``ghostap``."""
    parser = argparse.ArgumentParser(
        prog="ghostap",
        description="GhostAP — 飞书原生多工具 AI 研发控制面与执行网关",
    )
    parser.add_argument(
        "--validate",
        "--check-config",
        action="store_true",
        dest="validate",
        help="仅校验配置后退出，不启动服务",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        dest="version",
        help="显示 GhostAP 版本号",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="启动 GhostAP 飞书服务（默认行为）")

    validate_parser = subparsers.add_parser("validate", help="校验 .env 配置并退出")
    validate_parser.add_argument(
        "--check-config",
        action="store_true",
        help="与 validate 等效",
    )

    subparsers.add_parser("version", help="显示 GhostAP 版本号")

    init_parser = subparsers.add_parser("init", help="初始化 .env 配置文件（0600 权限）")
    init_parser.add_argument(
        "--dir",
        default=".",
        help="生成 .env 的目标目录（默认当前目录）",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="若目标 .env 已存在则强制覆盖",
    )

    doctor_parser = subparsers.add_parser("doctor", help="检查 Python / uv / Node.js / .env / 9 后端 CLI")
    doctor_parser.add_argument(
        "--dir",
        default=".",
        help="检查 .env 所在目录（默认当前目录）",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the ``ghostap`` executable."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version or args.command == "version":
        print(f"ghostap {_get_version()}")
        return 0

    if args.validate or args.command == "validate":
        from .main import main as run_main

        run_main(["--validate"])
        return 0

    if args.command == "init":
        return init_env(Path(args.dir), force=args.force)

    if args.command == "doctor":
        return run_doctor(Path(args.dir))

    from .main import main as run_main

    run_main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
