from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESTART_SCRIPT = ROOT / "restart.sh"


def test_restart_script_preheats_codex_acp_fallback_dependency():
    text = RESTART_SCRIPT.read_text(encoding="utf-8")

    assert "CODEX_ACP_NPM_PACKAGE=" in text
    assert "@agentclientprotocol/codex-acp@1.1.7" in text
    assert "PREPARE_CODEX_ACP=" in text
    assert "prepare_codex_acp_dependency()" in text
    assert 'npx --yes "$CODEX_ACP_NPM_PACKAGE" --version' in text
    assert 'npx --yes "$CODEX_ACP_NPM_PACKAGE" --help' not in text
    assert "prepare_codex_acp_dependency" in text.split("start_service() {", 1)[1]


@pytest.mark.parametrize(
    ("close_result", "expected_shutdown_events"),
    [
        (True, ["feishu-close", "locks-close"]),
        (False, ["feishu-close"]),
    ],
)
def test_application_startup_preheats_codex_model_capabilities_without_join(
    monkeypatch,
    close_result,
    expected_shutdown_events,
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
            return close_result

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
        staticmethod(lambda: events.append("locks-close")),
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
        *expected_shutdown_events,
    ]


def test_restart_script_syncs_python_and_prepares_platform_sandbox():
    text = RESTART_SCRIPT.read_text(encoding="utf-8")
    start_body = text.split("start_service() {", 1)[1]

    assert "GHOSTAP_SYNC_PYTHON_DEPENDENCIES" in text
    assert "uv sync --check --group dev" in text
    assert "uv sync --group dev" in text
    assert "venv_has_stale_entrypoint_shebang" in text
    assert "uv sync --group dev --reinstall" in text
    assert "prepare_python_dependencies || return 1" in start_body
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

    assert 'kill -0 "$target_pid" 2>/dev/null' in text
    assert 'pid_is_ghostap_service "$target_pid"' in text
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
                'unset GHOSTAP_LAUNCHCTL_LABEL; '
                'export GHOSTAP_RESTART_LIBRARY_ONLY=1; '
                'source "$1"; printf "%s\n" "$LAUNCHCTL_LABEL"',
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
        assert label.startswith("com.ghostap.local.")


def test_remote_restart_launches_python_gate_without_shared_worker_script(
    tmp_path,
):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "worker-args"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
uname() { echo Linux; }
snapshot_restart_generation() { printf '%s\n' AAAAAAAAAAAAAAAAAAAAAAAA; }
setsid() {
    printf '%s\n' "$@" > "$CAPTURE"
}
disown() { :; }
remote_restart
for _ in 1 2 3 4 5; do
    [ -s "$CAPTURE" ] && break
    sleep 0.05
done
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = capture.read_text(encoding="utf-8").splitlines()
    assert args[:6] == [
        "uv",
        "run",
        "--project",
        str(checkout),
        "python",
        "-m",
    ]
    assert args[6:8] == ["src.utils.restart_gate", "worker"]
    assert args[args.index("--project-dir") + 1] == str(checkout)
    assert "--gate-dir" not in args
    assert "--timeout" not in args
    assert args[args.index("--expected-generation") + 1] == "A" * 24
    assert args[args.index("--restart-script") + 1] == str(script)
    assert args[args.index("--log-file") + 1] == str(checkout / "logs.log")
    assert not (checkout / ".restart_worker.sh").exists()


def test_remote_restart_macos_uses_unique_launchctl_label_per_request(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "launchctl-args"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
uname() { echo Darwin; }
snapshot_restart_generation() { printf '%s\n' AAAAAAAAAAAAAAAAAAAAAAAA; }
launchctl() {
    printf '%s\n' "$*" >> "$CAPTURE"
}
disown() { :; }
remote_restart
remote_restart
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = capture.read_text(encoding="utf-8").splitlines()
    submits = [line for line in calls if line.startswith("submit -l ")]
    assert len(submits) == 2
    labels = [line.split()[2] for line in submits]
    assert labels[0] != labels[1]
    assert all(".restart." in label for label in labels)
    assert all("launch-wrapper" in line for line in submits)
    assert all("--launchd-label" in line for line in submits)
    assert all("--expected-generation AAAAAAAAAAAAAAAAAAAAAAAA" in line for line in submits)
    assert all(f"--project {checkout}" in line for line in submits)
    assert not any(line.startswith("remove ") for line in calls)


def test_remote_restart_preflight_failure_is_synchronous_and_does_not_detach(
    tmp_path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "events"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
uname() { echo Linux; }
snapshot_restart_generation() {
    printf '%s\n' snapshot-failed >> "$CAPTURE"
    return 75
}
setsid() {
    printf '%s\n' detached >> "$CAPTURE"
}
log_restart() { :; }
remote_restart
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert capture.read_text(encoding="utf-8").splitlines() == ["snapshot-failed"]
    assert "安全重启预检失败" in result.stdout


def test_macos_service_launch_uses_positional_argv_for_quoted_paths(tmp_path):
    checkout = tmp_path / "checkout with ' quote"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "launchctl-argv"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
command() {
    if [ "$1" = "-v" ] && [ "$2" = "setsid" ]; then
        return 1
    fi
    builtin command "$@"
}
launchctl() {
    if [ "$1" = "submit" ]; then
        printf '%s\n' "$@" > "$CAPTURE"
    fi
}
start_service_process append
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = capture.read_text(encoding="utf-8").splitlines()
    command_text = argv[argv.index("-lc") + 1]
    assert str(checkout) not in command_text
    assert "'$PROJECT_DIR'" not in command_text
    assert 'cd "$1"' in command_text
    assert str(checkout) in argv
    assert str(checkout / "logs.log") in argv


def test_start_service_publishes_generation_only_after_readiness(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "events"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
prepare_python_dependencies() { printf '%s\n' dependencies >> "$CAPTURE"; }
prepare_employee_sandbox_dependency() { :; }
prepare_codex_acp_dependency() { :; }
prepare_tui2acp_dependency() { :; }
log_restart() { :; }
start_service_process() {
    STARTED_PID=4242
    printf '%s\n' spawned >> "$CAPTURE"
}
get_running_pids() {
    printf '%s\n' 4242
}
wait_for_service_readiness() {
    printf '%s\n' ready >> "$CAPTURE"
    RUNNING_PIDS=4242
}
publish_restart_generation() {
    printf '%s\n' published >> "$CAPTURE"
}
sleep() { :; }
start_service
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "dependencies",
        "spawned",
        "ready",
        "published",
    ]


def test_start_service_cleans_spawned_process_when_readiness_fails(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "events"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
prepare_python_dependencies() { :; }
prepare_employee_sandbox_dependency() { :; }
prepare_codex_acp_dependency() { :; }
prepare_tui2acp_dependency() { :; }
log_restart() { :; }
start_service_process() {
    STARTED_PID=4242
    printf '%s\n' spawned >> "$CAPTURE"
}
wait_for_service_readiness() {
    printf '%s\n' readiness-failed >> "$CAPTURE"
    return 1
}
cleanup_failed_start() {
    printf 'cleanup:%s\n' "$1" >> "$CAPTURE"
}
publish_restart_generation() {
    printf '%s\n' unexpected-publish >> "$CAPTURE"
}
start_service
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "spawned",
        "readiness-failed",
        "cleanup:4242",
    ]
    assert "readiness" in result.stdout.lower()


def test_start_service_cleans_ready_process_when_generation_publish_fails(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "events"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
prepare_python_dependencies() { :; }
prepare_employee_sandbox_dependency() { :; }
prepare_codex_acp_dependency() { :; }
prepare_tui2acp_dependency() { :; }
log_restart() { :; }
start_service_process() { STARTED_PID=4242; }
wait_for_service_readiness() { RUNNING_PIDS=4242; }
publish_restart_generation() {
    printf '%s\n' publish-failed >> "$CAPTURE"
    return 1
}
cleanup_failed_start() {
    printf 'cleanup:%s\n' "$1" >> "$CAPTURE"
}
start_service
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "publish-failed",
        "cleanup:4242",
    ]


def test_forced_stop_is_degraded_and_process_group_is_only_killed_after_grace(
    tmp_path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "restart.sh"
    shutil.copy2(RESTART_SCRIPT, script)
    capture = tmp_path / "events"
    shell = r'''
export GHOSTAP_RESTART_LIBRARY_ONLY=1
export CAPTURE="$2"
source "$1"
TERM_GRACE_DELAY=0
printf '%s\n' 4242 > "$PID_FILE"
alive=1
pid_is_ghostap_service() { return 0; }
get_running_pids() { :; }
log_restart() { printf 'log:%s\n' "$*" >> "$CAPTURE"; }
kill() {
    if [ "$1" = "-0" ]; then
        [ "$alive" = "1" ]
        return
    fi
    printf 'kill:%s\n' "$*" >> "$CAPTURE"
    if [ "$1" = "-9" ]; then
        alive=0
    fi
}
sleep() { printf 'sleep:%s\n' "$1" >> "$CAPTURE"; }
stop_service
'''

    result = subprocess.run(
        ["bash", "-c", shell, "bash", str(script), str(capture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    events = capture.read_text(encoding="utf-8").splitlines()
    assert events.index("kill:4242") < events.index("kill:-9 4242")
    assert "kill:-- -4242" not in events
    assert "kill:-9 -- -4242" in events
    assert "强制" in result.stdout
    assert "✅ 服务已停止" not in result.stdout


def test_restart_script_has_no_generated_shared_worker_or_heredoc():
    text = RESTART_SCRIPT.read_text(encoding="utf-8")

    assert ".restart_worker.sh" not in text
    assert "WORKER_EOF" not in text
