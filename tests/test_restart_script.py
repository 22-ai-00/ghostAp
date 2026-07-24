from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RESTART_SCRIPT = ROOT / "restart.sh"


def test_restart_script_preheats_codex_acp_fallback_dependency():
    text = RESTART_SCRIPT.read_text(encoding="utf-8")

    assert "CODEX_ACP_NPM_PACKAGE=" in text
    assert "@agentclientprotocol/codex-acp@1.1.2" in text
    assert "PREPARE_CODEX_ACP=" in text
    assert "prepare_codex_acp_dependency()" in text
    assert 'npx --yes "$CODEX_ACP_NPM_PACKAGE" --version' in text
    assert 'npx --yes "$CODEX_ACP_NPM_PACKAGE" --help' not in text
    assert "prepare_codex_acp_dependency" in text.split("start_service() {", 1)[1]


def test_application_startup_preheats_codex_model_capabilities_without_join(
    monkeypatch,
):
    import src.acp.helper as acp_helper
    import src.coco_model as coco_model
    import src.main as main_module

    events = []
    preheat_calls = []

    class ThreadMustNotBeJoined:
        def join(self, *_args, **_kwargs):
            raise AssertionError("application startup must not join ACP preheat")

    class FakeCocoModelManager:
        def kickoff_preheat(self):
            events.append("coco-preheat")

    class FakeFeishuClient:
        def __init__(self, *, message_callback):
            self.message_callback = message_callback

        def start(self):
            events.append("feishu-start")

        def close(self):
            events.append("feishu-close")

    def fake_codex_preheat(tool_names, cwd):
        preheat_calls.append((tool_names, cwd))
        events.append("codex-preheat")
        return ThreadMustNotBeJoined()

    monkeypatch.setattr(acp_helper, "kickoff_acp_model_preheat", fake_codex_preheat)
    monkeypatch.setattr(
        coco_model,
        "get_coco_model_manager",
        lambda: FakeCocoModelManager(),
    )
    monkeypatch.setattr(main_module.os, "getcwd", lambda: "/repo")
    monkeypatch.setattr(
        main_module,
        "_load_feishu_runtime",
        lambda: (None, None, FakeFeishuClient),
    )
    monkeypatch.setattr(
        main_module.Application,
        "_install_signal_handlers",
        lambda _self: None,
    )
    monkeypatch.setattr(
        main_module.Application,
        "_shutdown_lock_managers",
        staticmethod(lambda: None),
    )

    app = object.__new__(main_module.Application)
    app.settings = SimpleNamespace(
        validate_feishu_config=lambda: True,
        app_id="test-app-id",
        sandbox_timeout=30,
        default_acp_tool=None,
        ttadk_preheat_enabled=False,
        ttadk_preheat_on_startup=False,
        acp_model_preheat_on_startup=True,
    )
    app.feishu_client = None

    app.run()

    assert preheat_calls == [(["codex"], "/repo")]
    assert events == [
        "coco-preheat",
        "codex-preheat",
        "feishu-start",
        "feishu-close",
    ]


def test_restart_script_syncs_python_and_prepares_platform_sandbox():
    text = RESTART_SCRIPT.read_text(encoding="utf-8")
    start_body = text.split("start_service() {", 1)[1]

    assert "GHOSTAP_SYNC_PYTHON_DEPENDENCIES" in text
    assert "uv sync --check --group dev" in text
    assert "uv sync --group dev" in text
    assert "venv_has_stale_entrypoint_shebang" in text
    assert "uv sync --group dev --reinstall" in text
    assert "prepare_python_dependencies || exit 1" in start_body
    assert "GHOSTAP_PREPARE_EMPLOYEE_SANDBOX" in text
    assert "prepare_employee_sandbox_dependency" in start_body
    assert "apt-get install -y bubblewrap" in text
    assert "dnf install -y bubblewrap" in text
    assert "pacman -S --needed --noconfirm bubblewrap" in text
    assert "pacman -Sy" not in text
    assert "/usr/bin/sandbox-exec" in text
    assert "mechanism=seatbelt" in text


def test_restart_script_syntax_is_valid():
    result = subprocess.run(
        ["bash", "-n", str(RESTART_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_restart_process_discovery_is_scoped_to_project_cwd():
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
source "$1"
uname() { echo Linux; }
ps() {
    if [ "$1" = "-axo" ]; then
        printf '%s\n' \
            '4242 uv run python -m src.main' \
            '4343 /tmp/other/.venv/bin/python -m src.main' \
            '4545 harmless uv run python -m src.main marker' \
            "4444 $PROJECT_DIR/.venv/bin/python -m src.main"
        return
    fi
    command ps "$@"
}
readlink() {
    case "$2" in
        */4444/cwd|*/4545/cwd) printf '%s\n' "$PROJECT_DIR" ;;
        */4444/exe|*/4545/exe|"$PYTHON_BIN") printf '%s\n' "$PYTHON_BIN" ;;
        *) printf '%s\n' '/tmp/other-checkout' ;;
    esac
}
get_running_pids
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(RESTART_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["4444"]


def test_restart_pid_file_is_validated_before_signalling():
    text = RESTART_SCRIPT.read_text(encoding="utf-8")

    assert 'kill -0 "$PID" 2>/dev/null && pid_is_ghostap_service "$PID"' in text
    assert "stale pid file ignored" in text
    assert "ps -axo pid=,command=" in text


def test_stale_pid_in_project_cwd_must_match_service_command():
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
source "$1"
uname() { echo Linux; }
readlink() { printf '%s\n' "$PROJECT_DIR"; }
ps() {
    if [ "$1" = "-p" ]; then
        printf '%s\n' "$FAKE_PROCESS_COMMAND"
        return
    fi
    command ps "$@"
}
FAKE_PROCESS_COMMAND='uv run pytest tests/'
if pid_is_ghostap_service 4242; then
    exit 9
fi
FAKE_PROCESS_COMMAND="$PROJECT_DIR/.venv/bin/python -m src.main"
pid_is_ghostap_service 4343
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(RESTART_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_default_launchctl_labels_are_isolated_per_checkout(tmp_path):
    labels: list[str] = []
    for name in ("checkout-a", "checkout-b"):
        checkout = tmp_path / name
        checkout.mkdir()
        script = checkout / "restart.sh"
        shutil.copy2(RESTART_SCRIPT, script)
        result = subprocess.run(
            [
                "bash",
                "-c",
                'unset GHOSTAP_LAUNCHCTL_LABEL '
                'GHOSTAP_RESTART_LAUNCHCTL_LABEL; '
                'export GHOSTAP_RESTART_LIBRARY_ONLY=1; '
                'source "$1"; printf "%s|%s\n" '
                '"$LAUNCHCTL_LABEL" "$RESTART_LAUNCHCTL_LABEL"',
                "bash",
                str(script),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        labels.append(result.stdout.strip())

    assert labels[0] != labels[1]
    for label in labels:
        service, worker = label.split("|")
        assert service.startswith("com.ghostap.local.")
        assert worker == f"{service}.restart"
