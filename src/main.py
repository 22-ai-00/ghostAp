import argparse
import logging
import os
import signal
import sys
import threading
from typing import Optional

try:
    from config import (
        ConfigurationError,
        SecuritySeverity,
        evaluate_security_posture,
        get_settings,
    )
    from utils.errors import get_error_detail
except ImportError:
    from .config import (
        ConfigurationError,
        SecuritySeverity,
        evaluate_security_posture,
        get_settings,
    )
    from .utils.errors import get_error_detail

logger = logging.getLogger(__name__)


def _load_feishu_runtime():
    """Load Feishu runtime dependencies only when the service actually starts."""
    try:
        from feishu.message_formatter import FeishuMessageFormatter as fmt
        from feishu.ws_client import EmojiReaction, FeishuWSClient
    except ImportError:
        from .feishu.message_formatter import FeishuMessageFormatter as fmt
        from .feishu.ws_client import EmojiReaction, FeishuWSClient
    return fmt, EmojiReaction, FeishuWSClient


def _format_duration(seconds: int) -> str:
    """Format seconds as Chinese-friendly '1800 秒（30 分钟）' with human-readable suffix."""
    import math

    if seconds <= 0:
        return "0 秒"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{seconds} 秒（{minutes} 分钟）"
    # Non-exact minutes: approximate with ceiling
    approx_min = math.ceil(seconds / 60)
    return f"{seconds} 秒（~{approx_min} 分钟）"


def _format_seconds(value: float) -> str:
    """Format integer-like second values with duration suffixes while preserving fractional timeouts."""
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:g} 秒"
    if seconds.is_integer():
        return _format_duration(int(seconds))
    return f"{seconds:g} 秒"


def _get_version() -> str:
    """Return package version via importlib.metadata, fallback to pyproject.toml."""
    try:
        from importlib.metadata import version
        return version("ghostap")
    except Exception:
        pass
    # Fallback: read from pyproject.toml
    try:
        import pathlib
        import tomllib

        pyproject_path = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            return data.get("project", {}).get("version", "")
    except Exception:
        pass
    return ""


def _print_validate_summary(settings):
    """Print structured --validate summary with version, groups, and units."""
    version = _get_version()
    if version:
        print(f"GhostAP v{version} 配置校验")
    else:
        print("GhostAP 配置校验")
    print()

    # Group 1: 会话超时
    print("[会话超时]")
    print(f"  CARD_SESSION_IDLE_TIMEOUT           = {_format_duration(settings.card.session_idle_timeout)}")
    print(f"  CARD_SESSION_IDLE_WARN_AT_REMAINING = {_format_duration(settings.card.session_idle_warn_at_remaining)}")
    print()

    # Group 2: 锁定参数
    print("[锁定参数]")
    print(f"  LOCK_UNDO_WINDOW_SECONDS = {_format_duration(settings.lock_undo_window_seconds)}")
    print()

    # Group 3: 高级参数
    print("[高级参数]")
    print(f"  CARD_DELIVERY_POOL_MAX_WORKERS = {settings.card.delivery_pool_max_workers} (threads)")
    print(f"  CARD_DELIVERY_API_TIMEOUT      = {_format_seconds(settings.card.delivery_api_timeout)}")
    print(f"  CARD_MAX_CHARS                 = {settings.card.max_chars} chars")
    print(f"  CARD_SESSION_LOCK_TTL          = {_format_duration(int(settings.card.session_lock_ttl))}")
    print(f"  CARD_SESSION_LOCK_MAX          = {settings.card.session_lock_max}")
    print()

    posture = evaluate_security_posture(settings)
    print("[安全姿态]")
    print(f"  INGRESS_ACCESS_MODE = {posture.ingress_mode.value}")
    print(f"  SHELL_ACCESS_MODE   = {posture.shell_mode.value}")
    print(
        "  EMPLOYEE_DEPARTMENT = "
        f"{'enabled' if posture.employee_department_enabled else 'disabled'}"
    )
    if posture.findings:
        for finding in posture.findings:
            print(
                f"  [{finding.severity.value}] {finding.code}: "
                f"{finding.message}"
            )
    else:
        print("  [info] no_security_findings")
    return posture


class Application:
    def __init__(self):
        self.settings = get_settings()
        self.feishu_client: Optional[object] = None
        self._shutdown_once = threading.Event()

    def _install_signal_handlers(self):
        """Ensure SIGTERM triggers graceful cleanup; ignore SIGHUP to survive
        terminal/SSH disconnects that previously killed the service mid-run.

        restart.sh uses SIGTERM; without a handler Python may exit immediately
        and skip finally blocks, leaving child agent processes orphaned.
        """

        def _handle_sigterm(signum, frame):  # pragma: no cover
            if self._shutdown_once.is_set():
                return
            self._shutdown_once.set()
            try:
                sig_name = signal.Signals(signum).name
            except Exception:
                logger.debug("_handle_sigterm: signal name lookup failed for %s", signum, exc_info=True)
                sig_name = str(signum)
            logger.info("收到终止信号 %s，开始优雅停机", sig_name)
            raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGTERM, _handle_sigterm)
        except Exception:
            # Some environments (non-main thread / restricted) may fail; best-effort.
            logger.debug("_install_signal_handlers: SIGTERM handler install failed", exc_info=True)

        # 忽略 SIGHUP：避免 SSH 会话/终端关闭意外终止长时间运行的引擎任务。
        # restart.sh 使用 SIGTERM 停机，不依赖 SIGHUP。
        try:
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except Exception:
            logger.debug("_install_signal_handlers: SIGHUP ignore failed", exc_info=True)

    def handle_message(self, message_id: str, chat_id: str, command: str, working_dir: Optional[str] = None):
        """Legacy callback — executes shell command directly via SandboxExecutor."""
        try:
            self.feishu_client._handler_ctx.handlers["system"].execute_shell_and_reply(
                message_id,
                chat_id,
                command,
                working_dir,
            )
        except Exception as e:
            logger.error("处理命令异常: %s", get_error_detail(e))
            try:
                fmt, EmojiReaction, _FeishuWSClient = _load_feishu_runtime()
                base = self.feishu_client._handler_ctx.handlers["coco"]
                base.add_reaction(message_id, EmojiReaction.on_error())
                base.reply_text(message_id, fmt.format_error(get_error_detail(e)))
            except Exception:
                logger.debug("failed to reply error message", exc_info=True)

    @staticmethod
    def _shutdown_lock_managers() -> None:
        """Best-effort shutdown of lock-manager singletons (stops daemon threads)."""
        try:
            from chat_lock import shutdown_if_active as _chat_shutdown
        except ImportError:
            from .chat_lock import shutdown_if_active as _chat_shutdown
        try:
            from repo_lock import shutdown_if_active as _repo_shutdown
        except ImportError:
            from .repo_lock import shutdown_if_active as _repo_shutdown

        _chat_shutdown()
        _repo_shutdown()

    def run(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        logger.info("=" * 50)
        logger.info("GhostAP - 飞书机器人Shell沙箱服务")
        logger.info("=" * 50)

        if not self.settings.validate_feishu_config():
            logger.error("配置错误: APP_ID 和 APP_SECRET 未配置或不完整，请参考 .env.example 模板配置 .env 文件")
            sys.exit(1)

        logger.info("APP_ID: %s...", self.settings.app_id[:8])
        logger.info("命令超时: %d秒", self.settings.sandbox_timeout)
        if self.settings.default_acp_tool:
            logger.info("默认 ACP 工具: %s", self.settings.default_acp_tool)
        else:
            logger.info("默认模式: 纯 Shell")

        # ACP model list preheat (background best-effort).
        try:
            if getattr(self.settings, "acp_model_preheat_on_startup", True):
                from .acp.helper import kickoff_acp_model_preheat
                from .coco_model import get_coco_model_manager

                get_coco_model_manager().kickoff_preheat()
                kickoff_acp_model_preheat(["codex"], cwd=os.getcwd())
        except Exception:
            logger.debug("Application.run: ACP model preheat failed", exc_info=True)

        self._install_signal_handlers()

        _fmt, _EmojiReaction, FeishuWSClient = _load_feishu_runtime()
        self.feishu_client = FeishuWSClient(message_callback=self.handle_message)

        logger.info("启动飞书长连接服务...")
        logger.info("支持的功能: Shell模式 | Coco模式 | 目录切换")

        try:
            self.feishu_client.start()
        except KeyboardInterrupt:
            logger.info("服务已停止")
        except Exception as e:
            logger.error("服务异常: %s", get_error_detail(e))
            sys.exit(1)
        finally:
            # The client first fences intake and drains task callbacks that may
            # still release or touch chat/repository locks.  Global lock
            # managers are leaf resources and must outlive those callbacks.
            client_drained = self.feishu_client is None
            try:
                if self.feishu_client:
                    client_drained = self.feishu_client.close() is not False
            except Exception:
                logger.debug("failed to close feishu client", exc_info=True)
                client_drained = False
            if client_drained:
                for _shutdown_fn in (
                    self._shutdown_lock_managers,
                ):
                    try:
                        _shutdown_fn()
                    except Exception:
                        logger.debug(
                            "lock manager shutdown error",
                            exc_info=True,
                        )
            else:
                logger.error(
                    "client work did not drain; preserving lock managers for "
                    "live callbacks"
                )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="GhostAP - 飞书机器人服务")
    parser.add_argument(
        "--validate", "--check-config",
        action="store_true",
        dest="validate",
        help="仅校验配置后退出，不启动服务",
    )
    args, _ = parser.parse_known_args(argv)

    if args.validate:
        try:
            settings = get_settings()
            if not settings.validate_feishu_config():
                sys.stderr.write(
                    f"{'=' * 40}\n[配置校验失败]\n"
                    "飞书应用配置不完整: APP_ID 和 APP_SECRET 不能为空\n"
                    f"{'=' * 40}\n"
                )
                sys.exit(1)
            posture = _print_validate_summary(settings)
            if not posture.is_valid:
                blocking_codes = ", ".join(
                    finding.code
                    for finding in posture.findings
                    if finding.severity is SecuritySeverity.BLOCKING
                )
                sys.stderr.write(
                    f"{'=' * 40}\n[配置校验失败]\n"
                    "安全姿态存在阻断项: "
                    f"{blocking_codes}\n{'=' * 40}\n"
                )
                sys.exit(1)
            print()
            print("配置校验通过")
            sys.exit(0)
        except ConfigurationError as e:
            sys.stderr.write(f"{'=' * 40}\n[配置校验失败]\n{e}\n{'=' * 40}\n")
            sys.exit(1)

    try:
        app = Application()
    except ConfigurationError as e:
        sys.stderr.write(f"{'=' * 40}\n[配置校验失败]\n{e}\n{'=' * 40}\n")
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
