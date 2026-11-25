from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = ROOT / "install.sh"
PYPROJECT_TOML = ROOT / "pyproject.toml"


def _create_isolated_sys_bin(
    tmp_path: Path,
    *,
    include_commands: tuple[str, ...] = (
        "awk",
        "basename",
        "bash",
        "cat",
        "chmod",
        "cksum",
        "cp",
        "date",
        "dirname",
        "env",
        "grep",
        "head",
        "ls",
        "mkdir",
        "mktemp",
        "pwd",
        "rm",
        "sh",
        "sleep",
        "tr",
        "uname",
    ),
) -> Path:
    """Create an isolated bin dir with essential POSIX helpers and no uv/git/curl/node."""
    sys_bin = tmp_path / "sys_bin"
    sys_bin.mkdir(parents=True, exist_ok=True)
    for name in include_commands:
        resolved = shutil.which(name)
        if resolved:
            link = sys_bin / name
            if not link.exists():
                link.symlink_to(resolved)
    return sys_bin


class TestGhostapCLI:
    """Unit tests for ``src.cli`` entry points and subcommands."""

    def test_version_flag_and_subcommand(self, capsys, monkeypatch):
        from src import cli

        monkeypatch.setattr(cli, "_get_version", lambda: "0.2.1")

        rc_flag = cli.main(["--version"])
        assert rc_flag == 0
        out_flag = capsys.readouterr().out.strip()
        assert out_flag == "ghostap 0.2.1"

        rc_sub = cli.main(["version"])
        assert rc_sub == 0
        out_sub = capsys.readouterr().out.strip()
        assert out_sub == "ghostap 0.2.1"

    def test_module_cli_version_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--version"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith("ghostap ")

    def test_run_and_default_delegate_to_main(self, monkeypatch):
        import src.main as main_mod
        from src import cli

        calls: list[list[str] | None] = []
        monkeypatch.setattr(main_mod, "main", lambda argv=None: calls.append(argv))

        assert cli.main([]) == 0
        assert cli.main(["run"]) == 0
        assert calls == [[], []]

    @pytest.mark.parametrize("argv", [["--validate"], ["--check-config"], ["validate"]])
    def test_validate_variants_delegate_to_main_validate(self, monkeypatch, argv):
        import src.main as main_mod
        from src import cli

        calls: list[list[str] | None] = []
        monkeypatch.setattr(main_mod, "main", lambda passed=None: calls.append(passed))

        assert cli.main(argv) == 0
        assert calls == [["--validate"]]

    def test_init_creates_env_with_0600_from_example_and_refuses_overwrite_without_force(
        self, tmp_path, capsys
    ):
        from src import cli

        example = tmp_path / ".env.example"
        example.write_text("APP_ID=example_id\nAPP_SECRET=example_secret\n", encoding="utf-8")

        rc1 = cli.main(["init", "--dir", str(tmp_path)])
        assert rc1 == 0
        env_file = tmp_path / ".env"
        assert env_file.is_file()
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
        assert "APP_ID=example_id" in env_file.read_text(encoding="utf-8")

        # Modify .env and widen perms to verify non-overwrite without --force
        env_file.write_text("APP_ID=custom_keep\nAPP_SECRET=keep_secret\n", encoding="utf-8")
        os.chmod(env_file, 0o644)

        rc2 = cli.main(["init", "--dir", str(tmp_path)])
        assert rc2 == 0
        assert "APP_ID=custom_keep" in env_file.read_text(encoding="utf-8")
        assert "已存在" in capsys.readouterr().out

        # --force overwrites and restores 0600 permissions
        rc3 = cli.main(["init", "--dir", str(tmp_path), "--force"])
        assert rc3 == 0
        assert "APP_ID=example_id" in env_file.read_text(encoding="utf-8")
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600

    def test_init_uses_builtin_template_when_env_example_missing(self, tmp_path, monkeypatch):
        from src import cli

        monkeypatch.setattr(cli, "_find_env_example", lambda _dir: None)

        rc = cli.main(["init", "--dir", str(tmp_path)])
        assert rc == 0
        env_file = tmp_path / ".env"
        assert env_file.is_file()
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
        text = env_file.read_text(encoding="utf-8")
        assert "APP_ID=your_app_id" in text
        assert "APP_SECRET=your_app_secret" in text

    def test_doctor_reports_prerequisites_and_all_nine_backends(
        self, tmp_path, capsys, monkeypatch
    ):
        from src import cli
        from src.mode.backends import VISIBLE_BACKEND_SPECS

        (tmp_path / ".env").write_text(
            "APP_ID=cli_real_app_id\nAPP_SECRET=cli_real_secret\n",
            encoding="utf-8",
        )

        available_bins = {
            "uv": "/usr/local/bin/uv",
            "node": "/usr/local/bin/node",
            "coco": "/usr/local/bin/coco",
            "claude": "/usr/local/bin/claude",
            "agy": "/usr/local/bin/agy",
        }
        monkeypatch.setattr(cli.shutil, "which", lambda name: available_bins.get(name))

        def fake_run(cmd, **_kwargs):
            binary = Path(cmd[0]).name
            if binary == "uv":
                return SimpleNamespace(returncode=0, stdout="uv 0.10.0\n", stderr="")
            if binary == "node":
                return SimpleNamespace(returncode=0, stdout="v22.5.0\n", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)

        rc = cli.main(["doctor", "--dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out

        assert "Python (>= 3.11)" in out
        assert "uv" in out
        assert "Node.js (>= 20.0.0, /wf)" in out
        assert "v22.5.0" in out
        assert "configured" in out
        assert "(3/9 detected)" in out

        for spec in VISIBLE_BACKEND_SPECS:
            assert spec.cli_name in out
            assert spec.display_name in out

    def test_doctor_flags_incomplete_env_and_old_node(self, tmp_path, capsys, monkeypatch):
        from src import cli

        (tmp_path / ".env").write_text(
            "APP_ID=your_app_id\nAPP_SECRET=\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            cli.shutil,
            "which",
            lambda name: "/usr/bin/node" if name == "node" else None,
        )
        monkeypatch.setattr(
            cli.subprocess,
            "run",
            lambda _cmd, **_kw: SimpleNamespace(returncode=0, stdout="v18.19.0\n", stderr=""),
        )

        rc = cli.main(["doctor", "--dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "incomplete: APP_ID, APP_SECRET" in out
        assert "WARN" in out
        assert "(0/9 detected)" in out


class TestWheelPackaging:
    """Verify ``pyproject.toml`` metadata and built wheel contents."""

    def test_pyproject_preserves_description_and_declares_cli_metadata(self):
        import tomllib

        raw = PYPROJECT_TOML.read_text(encoding="utf-8")
        assert 'description = "飞书原生多工具 AI 研发控制面与执行网关"' in raw

        data = tomllib.loads(raw)
        project = data["project"]
        assert project["readme"] == "README.md"
        assert project["license"] == {"text": "MIT"}
        assert project["scripts"]["ghostap"] == "src.cli:main"
        assert project["urls"]["Homepage"] == "https://github.com/22-ai-00/ghostAp"
        assert project["urls"]["Repository"] == "https://github.com/22-ai-00/ghostAp"
        assert project["urls"]["Issues"].startswith("https://github.com/22-ai-00/ghostAp")
        assert project["urls"]["Changelog"].startswith("https://github.com/22-ai-00/ghostAp")
        assert project["keywords"]
        assert project["classifiers"]

    def test_uv_build_wheel_contains_cli_and_workflow_js_runtime(self, tmp_path):
        dist_dir = tmp_path / "dist"
        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        wheels = list(dist_dir.glob("ghostap-*.whl"))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0]) as zf:
            names = set(zf.namelist())
            assert "src/cli.py" in names
            assert "src/main.py" in names
            assert "src/workflow_engine/runtime/runtime.js" in names
            assert "src/workflow_engine/runtime/package.json" in names
            entry_points_candidates = [n for n in names if n.endswith("entry_points.txt")]
            assert len(entry_points_candidates) == 1
            entry_points = zf.read(entry_points_candidates[0]).decode("utf-8")
            assert "ghostap = src.cli:main" in entry_points


class TestInstallScript:
    """Tests for standalone ``install.sh`` across Linux/macOS and all 3 modes."""

    def test_install_script_syntax_and_help(self):
        syntax = subprocess.run(
            ["bash", "-n", str(INSTALL_SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr
        assert os.access(INSTALL_SCRIPT, os.X_OK)

        help_run = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert help_run.returncode == 0, help_run.stderr
        assert "--mode" in help_run.stdout
        assert "--wheel" in help_run.stdout

    def test_piped_help_and_unknown_flags(self):
        piped = subprocess.run(
            ["bash", "-s", "--", "--help"],
            input=INSTALL_SCRIPT.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            check=False,
        )
        assert piped.returncode == 0, piped.stderr
        assert "GhostAP" in piped.stdout

        bad_flag = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--unknown-flag"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert bad_flag.returncode != 0
        assert "未知参数" in bad_flag.stderr

        bad_mode = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--mode", "docker"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert bad_mode.returncode != 0
        assert "无效的 --mode" in bad_mode.stderr

    def test_default_install_dir_detection_inside_and_outside_checkout(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        res_inside = subprocess.run(
            [
                "bash",
                "-c",
                'export GHOSTAP_INSTALL_LIBRARY_ONLY=1; source "$1"; resolve_default_install_dir',
                "bash",
                str(INSTALL_SCRIPT),
            ],
            cwd=str(ROOT),
            env={**os.environ, "HOME": str(home_dir)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_inside.returncode == 0, res_inside.stderr
        assert Path(res_inside.stdout.strip()).resolve() == ROOT.resolve()

        res_outside = subprocess.run(
            [
                "bash",
                "-c",
                'export GHOSTAP_INSTALL_LIBRARY_ONLY=1; source "$1"; resolve_default_install_dir',
                "bash",
                str(INSTALL_SCRIPT),
            ],
            cwd=str(outside),
            env={**os.environ, "HOME": str(home_dir)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_outside.returncode == 0, res_outside.stderr
        assert res_outside.stdout.strip() == str(home_dir / "ghostAp")

    @pytest.mark.parametrize("os_name", ["Linux", "Darwin"])
    def test_source_mode_clones_syncs_and_initializes_env_when_checkout_absent(
        self, tmp_path, os_name
    ):
        sys_bin = _create_isolated_sys_bin(tmp_path)
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        target_dir = tmp_path / "installed_ghostap"
        capture = tmp_path / "calls.log"

        # Mock uname, curl (installs fake uv), git (creates checkout with .env.example)
        (sys_bin / "uname").unlink(missing_ok=True)
        (sys_bin / "uname").write_text(f"#!/bin/sh\necho {os_name}\n", encoding="utf-8")
        (sys_bin / "uname").chmod(0o755)

        curl_script = f"""#!/bin/sh
echo "curl:$*" >> "{capture}"
cat <<'EOS'
mkdir -p "$HOME/.local/bin"
cat <<'EUV' > "$HOME/.local/bin/uv"
#!/bin/sh
echo "uv:$*" >> "{capture}"
exit 0
EUV
chmod +x "$HOME/.local/bin/uv"
EOS
"""
        (sys_bin / "curl").write_text(curl_script, encoding="utf-8")
        (sys_bin / "curl").chmod(0o755)

        git_script = f"""#!/bin/sh
echo "git:$*" >> "{capture}"
dest=""
for arg in "$@"; do
    dest="$arg"
done
mkdir -p "$dest"
printf 'name = "ghostap"\\n' > "$dest/pyproject.toml"
printf 'APP_ID=your_app_id\\nAPP_SECRET=your_app_secret\\n' > "$dest/.env.example"
"""
        (sys_bin / "git").write_text(git_script, encoding="utf-8")
        (sys_bin / "git").chmod(0o755)

        result = subprocess.run(
            ["bash", "-s", "--", "--mode", "source", "--dir", str(target_dir), "--branch", "main"],
            input=INSTALL_SCRIPT.read_text(encoding="utf-8"),
            cwd=str(tmp_path),
            env={
                "PATH": str(sys_bin),
                "HOME": str(home_dir),
                "TMPDIR": str(tmp_path),
            },
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert any(line.startswith("curl:") for line in calls)
        assert any(
            line == f"git:clone --branch main https://github.com/22-ai-00/ghostAp.git {target_dir}"
            for line in calls
        )
        assert "uv:python find >=3.11" in calls
        assert "uv:sync --group dev --compile-bytecode" in calls

        env_file = target_dir / ".env"
        assert env_file.is_file()
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
        assert "./restart.sh start" in result.stdout
        assert "/setadmin" in result.stdout

    def test_source_mode_fails_cleanly_without_git(self, tmp_path):
        sys_bin = _create_isolated_sys_bin(tmp_path)
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        target_dir = tmp_path / "missing_repo"

        (sys_bin / "uv").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (sys_bin / "uv").chmod(0o755)

        result = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--mode", "source", "--dir", str(target_dir)],
            cwd=str(tmp_path),
            env={"PATH": str(sys_bin), "HOME": str(home_dir)},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "git" in result.stderr

    def test_source_mode_no_dev_skips_dev_group_in_existing_checkout(self, tmp_path):
        sys_bin = _create_isolated_sys_bin(tmp_path)
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        checkout = tmp_path / "existing_checkout"
        checkout.mkdir()
        (checkout / "pyproject.toml").write_text('name = "ghostap"\n', encoding="utf-8")
        (checkout / ".env.example").write_text("APP_ID=x\n", encoding="utf-8")
        capture = tmp_path / "uv.log"

        (sys_bin / "uv").write_text(
            f'#!/bin/sh\necho "uv:$*" >> "{capture}"\nexit 0\n',
            encoding="utf-8",
        )
        (sys_bin / "uv").chmod(0o755)

        result = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--mode", "source", "--no-dev"],
            cwd=str(checkout),
            env={"PATH": str(sys_bin), "HOME": str(home_dir)},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert "uv:sync --compile-bytecode" in calls
        assert not any("--group dev" in line for line in calls)

    def test_tool_mode_installs_and_verifies_ghostap(self, tmp_path):
        sys_bin = _create_isolated_sys_bin(tmp_path)
        home_dir = tmp_path / "home"
        tool_bin = home_dir / ".local" / "bin"
        tool_bin.mkdir(parents=True)
        capture = tmp_path / "uv.log"

        uv_script = f"""#!/bin/sh
echo "uv:$*" >> "{capture}"
if [ "$1" = "tool" ] && [ "$2" = "install" ]; then
    cat <<'EOC' > "{tool_bin}/ghostap"
#!/bin/sh
echo "ghostap 0.2.1"
EOC
    chmod +x "{tool_bin}/ghostap"
fi
exit 0
"""
        (sys_bin / "uv").write_text(uv_script, encoding="utf-8")
        (sys_bin / "uv").chmod(0o755)

        # Outside checkout -> installs from git+repo
        outside = tmp_path / "outside"
        outside.mkdir()
        result = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--mode", "tool", "--branch", "main"],
            cwd=str(outside),
            env={"PATH": str(sys_bin), "HOME": str(home_dir)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert "uv:tool install --force git+https://github.com/22-ai-00/ghostAp.git@main" in calls
        assert "ghostap 0.2.1" in result.stdout

    def test_wheel_mode_supports_local_file_and_release_api_download(self, tmp_path):
        sys_bin = _create_isolated_sys_bin(tmp_path)
        home_dir = tmp_path / "home"
        tool_bin = home_dir / ".local" / "bin"
        tool_bin.mkdir(parents=True)
        capture = tmp_path / "calls.log"

        uv_script = f"""#!/bin/sh
echo "uv:$*" >> "{capture}"
if [ "$1" = "tool" ] && [ "$2" = "install" ]; then
    cat <<'EOC' > "{tool_bin}/ghostap"
#!/bin/sh
echo "ghostap 0.2.1"
EOC
    chmod +x "{tool_bin}/ghostap"
fi
exit 0
"""
        (sys_bin / "uv").write_text(uv_script, encoding="utf-8")
        (sys_bin / "uv").chmod(0o755)

        # 1. Local wheel file
        local_whl = tmp_path / "ghostap-0.2.1-py3-none-any.whl"
        local_whl.write_bytes(b"wheel-bytes")

        res_local = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--mode", "wheel", "--wheel", str(local_whl)],
            cwd=str(tmp_path),
            env={"PATH": str(sys_bin), "HOME": str(home_dir), "TMPDIR": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_local.returncode == 0, res_local.stderr
        assert f"uv:tool install --force {local_whl}" in capture.read_text(encoding="utf-8").splitlines()

        # 2. Latest release discovery via mocked curl + python3
        capture.unlink()
        py_link = sys_bin / "python3"
        py_link.symlink_to(sys.executable)
        curl_script = f"""#!/bin/sh
echo "curl:$*" >> "{capture}"
out=""
url=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) out="$2"; shift 2 ;;
        -*) shift ;;
        *) url="$1"; shift ;;
    esac
done
if [ -n "$out" ]; then
    printf 'fake-wheel' > "$out"
else
    printf '{{"assets": [{{"browser_download_url": "https://example.com/ghostap-0.2.1-py3-none-any.whl"}}]}}'
fi
"""
        (sys_bin / "curl").write_text(curl_script, encoding="utf-8")
        (sys_bin / "curl").chmod(0o755)

        res_release = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--mode", "wheel"],
            cwd=str(tmp_path),
            env={
                "PATH": str(sys_bin),
                "HOME": str(home_dir),
                "TMPDIR": str(tmp_path),
                "GHOSTAP_RELEASE_API_URL": "https://api.example.test/releases/latest",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_release.returncode == 0, res_release.stderr
        calls = capture.read_text(encoding="utf-8").splitlines()
        assert any("https://api.example.test/releases/latest" in line for line in calls)
        assert any("https://example.com/ghostap-0.2.1-py3-none-any.whl" in line for line in calls)
        assert any(
            line.startswith("uv:tool install --force ") and line.endswith("ghostap-0.2.1-py3-none-any.whl")
            for line in calls
        )

        # 3. Direct URL --wheel
        capture.unlink()
        res_url = subprocess.run(
            [
                "bash",
                str(INSTALL_SCRIPT),
                "--mode",
                "wheel",
                "--wheel",
                "https://mirror.example.test/custom-ghostap.whl",
            ],
            cwd=str(tmp_path),
            env={"PATH": str(sys_bin), "HOME": str(home_dir), "TMPDIR": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_url.returncode == 0, res_url.stderr
        url_calls = capture.read_text(encoding="utf-8").splitlines()
        assert any("https://mirror.example.test/custom-ghostap.whl" in line for line in url_calls)
        assert any(
            line.startswith("uv:tool install --force ") and line.endswith("custom-ghostap.whl")
            for line in url_calls
        )

    def test_install_script_uv_fallback_chain_and_disable_flag(self, tmp_path):
        sys_bin = _create_isolated_sys_bin(tmp_path)
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        capture = tmp_path / "events.log"

        disabled = subprocess.run(
            [
                "bash",
                "-c",
                'export GHOSTAP_INSTALL_LIBRARY_ONLY=1 GHOSTAP_AUTO_INSTALL_UV=0; source "$1"; ensure_uv_available',
                "bash",
                str(INSTALL_SCRIPT),
            ],
            env={"PATH": str(sys_bin), "HOME": str(home_dir)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert disabled.returncode != 0
        assert "GHOSTAP_AUTO_INSTALL_UV=0" in disabled.stderr

        (sys_bin / "curl").write_text(f'#!/bin/sh\necho curl-fail >> "{capture}"\nexit 1\n', encoding="utf-8")
        (sys_bin / "curl").chmod(0o755)
        (sys_bin / "wget").write_text(f'#!/bin/sh\necho wget-fail >> "{capture}"\nexit 1\n', encoding="utf-8")
        (sys_bin / "wget").chmod(0o755)
        (sys_bin / "python3").write_text(
            f"""#!/bin/sh
echo "python3:$*" >> "{capture}"
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
    exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
    mkdir -p "$HOME/.local/bin"
    printf '#!/bin/sh\\necho uv\\n' > "$HOME/.local/bin/uv"
    chmod +x "$HOME/.local/bin/uv"
    exit 0
fi
exit 1
""",
            encoding="utf-8",
        )
        (sys_bin / "python3").chmod(0o755)

        fallback = subprocess.run(
            [
                "bash",
                "-c",
                'export GHOSTAP_INSTALL_LIBRARY_ONLY=1; source "$1"; ensure_uv_available && command -v uv',
                "bash",
                str(INSTALL_SCRIPT),
            ],
            env={"PATH": str(sys_bin), "HOME": str(home_dir)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert fallback.returncode == 0, fallback.stderr
        events = capture.read_text(encoding="utf-8").splitlines()
        assert events == [
            "curl-fail",
            "wget-fail",
            "python3:-m pip --version",
            "python3:-m pip install --user uv",
        ]
