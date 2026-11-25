#!/bin/bash

# GhostAP standalone one-click installer for Linux and macOS.
# Supports piped execution (`curl -fsSL ... | bash`) and local execution (`./install.sh`).

DEFAULT_REPO_URL="https://github.com/22-ai-00/ghostAp.git"
DEFAULT_UV_INSTALLER_URL="https://astral.sh/uv/install.sh"
DEFAULT_RELEASE_API_URL="https://api.github.com/repos/22-ai-00/ghostAp/releases/latest"

INSTALL_MODE="source"
TARGET_DIR=""
TARGET_DIR_EXPLICIT=0
WHEEL_TARGET=""
REPO_URL="${GHOSTAP_REPO_URL:-$DEFAULT_REPO_URL}"
BRANCH=""
DEV_GROUP=1
AUTO_INSTALL_UV="${GHOSTAP_AUTO_INSTALL_UV:-1}"
UV_INSTALLER_URL="${GHOSTAP_UV_INSTALLER_URL:-$DEFAULT_UV_INSTALLER_URL}"

print_usage() {
    cat <<'EOF'
GhostAP 一键安装脚本 (Linux & macOS)

用法:
  ./install.sh [选项]
  curl -fsSL https://raw.githubusercontent.com/22-ai-00/ghostAp/main/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/22-ai-00/ghostAp/main/install.sh | bash -s -- [选项]

选项:
  --mode <source|tool|wheel>  安装模式（默认: source）
                                source: 克隆/同步源码并用 uv sync 安装依赖
                                tool:   通过 uv tool install 安装全局 ghostap CLI
                                wheel:  通过本地或远程 .whl 包安装全局 ghostap CLI
  --dir <target>              源码或工作目录（默认: 当前 GhostAP 仓库，否则 $HOME/ghostAp）
  --wheel <path-or-url>       指定 .whl 文件路径或 URL（仅 --mode wheel 使用，缺省拉取最新 Release）
  --repo <git-url>            Git 仓库地址（默认: https://github.com/22-ai-00/ghostAp.git）
  --branch <branch>           指定 Git 分支或标签（可选）
  --no-dev                    source 模式下跳过 dev 依赖组
  -h, --help                  显示帮助信息并退出
EOF
}

is_ghostap_checkout() {
    local dir="$1"
    [ -f "$dir/pyproject.toml" ] || return 1
    grep -Eq '^[[:space:]]*name[[:space:]]*=[[:space:]]*"ghostap"' "$dir/pyproject.toml" 2>/dev/null
}

resolve_default_install_dir() {
    if is_ghostap_checkout "$PWD"; then
        printf '%s\n' "$PWD"
    else
        printf '%s\n' "${GHOSTAP_INSTALL_DIR:-$HOME/ghostAp}"
    fi
}

parse_args() {
    INSTALL_MODE="source"
    TARGET_DIR=""
    TARGET_DIR_EXPLICIT=0
    WHEEL_TARGET=""
    REPO_URL="${GHOSTAP_REPO_URL:-$DEFAULT_REPO_URL}"
    BRANCH=""
    DEV_GROUP=1

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --mode)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "❌ --mode 需要指定参数: source | tool | wheel" >&2
                    return 2
                fi
                INSTALL_MODE="$2"
                shift 2
                ;;
            --mode=*)
                INSTALL_MODE="${1#*=}"
                shift
                ;;
            --dir)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "❌ --dir 需要指定目标路径" >&2
                    return 2
                fi
                TARGET_DIR="$2"
                TARGET_DIR_EXPLICIT=1
                shift 2
                ;;
            --dir=*)
                TARGET_DIR="${1#*=}"
                TARGET_DIR_EXPLICIT=1
                shift
                ;;
            --wheel)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "❌ --wheel 需要指定文件路径或 URL" >&2
                    return 2
                fi
                WHEEL_TARGET="$2"
                shift 2
                ;;
            --wheel=*)
                WHEEL_TARGET="${1#*=}"
                shift
                ;;
            --repo)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "❌ --repo 需要指定 Git 仓库 URL" >&2
                    return 2
                fi
                REPO_URL="$2"
                shift 2
                ;;
            --repo=*)
                REPO_URL="${1#*=}"
                shift
                ;;
            --branch)
                if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                    echo "❌ --branch 需要指定分支名" >&2
                    return 2
                fi
                BRANCH="$2"
                shift 2
                ;;
            --branch=*)
                BRANCH="${1#*=}"
                shift
                ;;
            --no-dev)
                DEV_GROUP=0
                shift
                ;;
            -h|--help)
                print_usage
                return 3
                ;;
            *)
                echo "❌ 未知参数: $1" >&2
                print_usage >&2
                return 2
                ;;
        esac
    done

    case "$INSTALL_MODE" in
        source|tool|wheel) ;;
        *)
            echo "❌ 无效的 --mode: $INSTALL_MODE（可选: source, tool, wheel）" >&2
            return 2
            ;;
    esac

    if [ -z "$TARGET_DIR" ]; then
        TARGET_DIR="$(resolve_default_install_dir)"
    fi
    return 0
}

prepend_path_if_missing() {
    local dir="$1"
    [ -d "$dir" ] || return 1
    case ":$PATH:" in
        *":$dir:"*) ;;
        *) export PATH="$dir:$PATH" ;;
    esac
    return 0
}

refresh_tool_bin_path() {
    local candidate_dirs=()
    if [ -n "${UV_TOOL_BIN_DIR:-}" ]; then
        candidate_dirs+=("$UV_TOOL_BIN_DIR")
    fi
    if [ -n "${HOME:-}" ]; then
        candidate_dirs+=(
            "$HOME/.local/bin"
            "$HOME/.cargo/bin"
        )
        local py_user_bin=""
        for py_user_bin in "$HOME"/Library/Python/*/bin; do
            if [ -d "$py_user_bin" ]; then
                candidate_dirs+=("$py_user_bin")
            fi
        done
    fi
    candidate_dirs+=(
        "/opt/homebrew/bin"
        "/usr/local/bin"
    )

    local dir=""
    for dir in "${candidate_dirs[@]}"; do
        if [ -x "$dir/uv" ] || [ -x "$dir/ghostap" ]; then
            prepend_path_if_missing "$dir" || true
        fi
    done
}

refresh_uv_path() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    refresh_tool_bin_path
    command -v uv >/dev/null 2>&1
}

ensure_uv_available() {
    refresh_uv_path
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi

    if [ "$AUTO_INSTALL_UV" = "0" ]; then
        echo "❌ 未找到 uv，且已禁用自动安装（GHOSTAP_AUTO_INSTALL_UV=0）" >&2
        return 1
    fi

    echo "📦 未检测到 uv，正在自动安装 uv..."
    local installed=0
    if command -v curl >/dev/null 2>&1; then
        if (set -o pipefail; curl -LsSf "$UV_INSTALLER_URL" | env UV_NO_MODIFY_PATH=1 sh); then
            installed=1
        fi
    fi
    if [ "$installed" = "0" ] && command -v wget >/dev/null 2>&1; then
        if (set -o pipefail; wget -qO- "$UV_INSTALLER_URL" | env UV_NO_MODIFY_PATH=1 sh); then
            installed=1
        fi
    fi
    if [ "$installed" = "0" ] && command -v brew >/dev/null 2>&1; then
        if brew install uv; then
            installed=1
        fi
    fi
    if [ "$installed" = "0" ] && command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
        if python3 -m pip install --user uv; then
            installed=1
        fi
    fi

    refresh_uv_path || true
    if command -v uv >/dev/null 2>&1; then
        echo "✅ uv 自动安装完成: $(command -v uv)"
        return 0
    fi

    echo "❌ uv 自动安装失败，请参考 https://docs.astral.sh/uv/getting-started/installation/" >&2
    return 1
}

ensure_python_available() {
    if uv python find '>=3.11' >/dev/null 2>&1; then
        return 0
    fi
    echo "🐍 未检测到 Python >= 3.11，正在通过 uv 安装 Python 3.12..."
    if ! uv python install 3.12; then
        echo "❌ Python >= 3.11 安装失败" >&2
        return 1
    fi
    uv python find '>=3.11' >/dev/null 2>&1
}

init_env_file() {
    local dir="$1"
    local env_file="$dir/.env"
    local example_file="$dir/.env.example"

    if [ -f "$env_file" ]; then
        echo "ℹ️  保留已有配置文件: $env_file"
        return 0
    fi

    if [ -f "$example_file" ]; then
        cp "$example_file" "$env_file" || return 1
        chmod 600 "$env_file" || return 1
        echo "✅ 已初始化配置文件: $env_file (权限 0600)"
        return 0
    fi

    if [ -x "$dir/.venv/bin/ghostap" ]; then
        "$dir/.venv/bin/ghostap" init --dir "$dir"
        return $?
    fi

    if command -v ghostap >/dev/null 2>&1; then
        ghostap init --dir "$dir"
        return $?
    fi

    echo "⚠️  未找到 .env.example，请稍后运行 ghostap init 创建 .env"
    return 0
}

ensure_source_checkout() {
    local target="$1"
    if [ -f "$target/pyproject.toml" ]; then
        return 0
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "❌ 未找到 git，无法克隆 GhostAP 仓库，请先安装 git 后重试" >&2
        return 1
    fi

    if [ -e "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "❌ 目标目录已存在且非空: $target" >&2
        return 1
    fi

    local parent_dir
    parent_dir="$(dirname "$target")"
    mkdir -p "$parent_dir" || return 1

    echo "📥 正在克隆 GhostAP 到 $target ..."
    if [ -n "$BRANCH" ]; then
        git clone --branch "$BRANCH" "$REPO_URL" "$target" || return 1
    else
        git clone "$REPO_URL" "$target" || return 1
    fi
}

install_source_mode() {
    ensure_source_checkout "$TARGET_DIR" || return 1

    echo "📦 正在同步 Python 依赖并预编译字节码..."
    if [ "$DEV_GROUP" = "1" ]; then
        (cd "$TARGET_DIR" && uv sync --group dev --compile-bytecode) || return 1
    else
        (cd "$TARGET_DIR" && uv sync --compile-bytecode) || return 1
    fi

    init_env_file "$TARGET_DIR" || return 1

    echo ""
    echo "✅ GhostAP 源码模式安装完成: $TARGET_DIR"
    echo "接下来只需 3 步:"
    echo "  1. 编辑配置: cd \"$TARGET_DIR\" && vim .env  # 填入 APP_ID 与 APP_SECRET"
    echo "  2. 启动服务: ./restart.sh start"
    echo "  3. 绑定管理员: 在飞书私聊 Bot 发送 /setadmin"
}

verify_ghostap_cli() {
    refresh_tool_bin_path
    if ! command -v ghostap >/dev/null 2>&1; then
        echo "❌ 未在 PATH 中找到已安装的 ghostap 命令，请将 \$HOME/.local/bin 加入 PATH" >&2
        return 1
    fi
    ghostap --version
}

install_tool_mode() {
    local tool_source=""
    if [ "$TARGET_DIR_EXPLICIT" = "1" ] || [ -f "$TARGET_DIR/pyproject.toml" ]; then
        if [ ! -f "$TARGET_DIR/pyproject.toml" ]; then
            echo "❌ 指定目录缺少 pyproject.toml: $TARGET_DIR" >&2
            return 1
        fi
        tool_source="$TARGET_DIR"
    else
        local git_spec="git+${REPO_URL}"
        if [ -n "$BRANCH" ]; then
            git_spec="${git_spec}@${BRANCH}"
        fi
        tool_source="$git_spec"
    fi

    echo "🔧 正在通过 uv tool install 安装 ghostap ($tool_source)..."
    uv tool install --force "$tool_source" || return 1
    verify_ghostap_cli || return 1

    echo ""
    echo "✅ GhostAP CLI 工具模式安装完成"
    echo "接下来只需 3 步:"
    echo "  1. 初始化配置: ghostap init && vim .env  # 填入 APP_ID 与 APP_SECRET"
    echo "  2. 环境体检与启动: ghostap doctor && ghostap --validate && ghostap"
    echo "  3. 绑定管理员: 在飞书私聊 Bot 发送 /setadmin"
}

download_url_to_file() {
    local url="$1"
    local dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$dest" && return 0
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -qO "$dest" "$url" && return 0
    fi
    echo "❌ 下载失败（需要 curl 或 wget）: $url" >&2
    return 1
}

fetch_url_stdout() {
    local url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" && return 0
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -qO- "$url" && return 0
    fi
    echo "❌ 获取元数据失败（需要 curl 或 wget）: $url" >&2
    return 1
}

derive_release_api_url() {
    if [ -n "${GHOSTAP_RELEASE_API_URL:-}" ]; then
        printf '%s\n' "$GHOSTAP_RELEASE_API_URL"
        return 0
    fi
    case "$REPO_URL" in
        https://github.com/*|http://github.com/*)
            local slug="${REPO_URL#http*://github.com/}"
            slug="${slug%.git}"
            slug="${slug%/}"
            printf 'https://api.github.com/repos/%s/releases/latest\n' "$slug"
            return 0
            ;;
        *)
            printf '%s\n' "$DEFAULT_RELEASE_API_URL"
            return 0
            ;;
    esac
}

extract_wheel_url_from_release_json() {
    local json_payload="$1"
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$json_payload" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for asset in data.get("assets") or []:
    url = str(asset.get("browser_download_url") or "")
    if url.endswith(".whl"):
        print(url)
        sys.exit(0)
sys.exit(1)
' && return 0
    fi
    printf '%s\n' "$json_payload" | grep -Eo 'https://[^"[:space:]]+\.whl' | head -n 1
}

resolve_wheel_target() {
    local requested="$1"
    local temp_dir="$2"

    if [ -n "$requested" ]; then
        case "$requested" in
            http://*|https://*)
                local filename
                filename="$(basename "${requested%%\?*}")"
                case "$filename" in
                    *.whl) ;;
                    *) filename="ghostap-latest-py3-none-any.whl" ;;
                esac
                local local_whl="$temp_dir/$filename"
                echo "📥 正在下载 wheel: $requested" >&2
                download_url_to_file "$requested" "$local_whl" || return 1
                printf '%s\n' "$local_whl"
                return 0
                ;;
            *)
                if [ ! -f "$requested" ]; then
                    echo "❌ 未找到指定的 wheel 文件: $requested" >&2
                    return 1
                fi
                printf '%s\n' "$requested"
                return 0
                ;;
        esac
    fi

    local api_url
    api_url="$(derive_release_api_url)"
    echo "🔎 正在查询最新 Release wheel: $api_url" >&2
    local release_json=""
    release_json="$(fetch_url_stdout "$api_url")" || return 1
    local wheel_url=""
    wheel_url="$(extract_wheel_url_from_release_json "$release_json")" || true
    if [ -z "$wheel_url" ]; then
        echo "❌ 最新 Release 中未找到 .whl 资源，请使用 --wheel <path-or-url> 或 --mode source" >&2
        return 1
    fi

    local filename
    filename="$(basename "${wheel_url%%\?*}")"
    local downloaded_whl="$temp_dir/$filename"
    echo "📥 正在下载 wheel: $wheel_url" >&2
    download_url_to_file "$wheel_url" "$downloaded_whl" || return 1
    printf '%s\n' "$downloaded_whl"
}

install_wheel_mode() {
    local temp_dir=""
    temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/ghostap-wheel.XXXXXX")" || return 1

    local resolved_wheel=""
    if ! resolved_wheel="$(resolve_wheel_target "$WHEEL_TARGET" "$temp_dir")"; then
        rm -rf "$temp_dir"
        return 1
    fi

    echo "📦 正在通过 wheel 安装 ghostap: $resolved_wheel"
    if ! uv tool install --force "$resolved_wheel"; then
        rm -rf "$temp_dir"
        return 1
    fi
    rm -rf "$temp_dir"

    verify_ghostap_cli || return 1

    echo ""
    echo "✅ GhostAP Wheel 模式安装完成"
    echo "接下来只需 3 步:"
    echo "  1. 初始化配置: ghostap init && vim .env  # 填入 APP_ID 与 APP_SECRET"
    echo "  2. 环境体检与启动: ghostap doctor && ghostap --validate && ghostap"
    echo "  3. 绑定管理员: 在飞书私聊 Bot 发送 /setadmin"
}

main() {
    local parse_rc=0
    parse_args "$@" || parse_rc=$?
    if [ "$parse_rc" = "3" ]; then
        return 0
    elif [ "$parse_rc" != "0" ]; then
        return "$parse_rc"
    fi

    ensure_uv_available || return 1
    ensure_python_available || return 1

    case "$INSTALL_MODE" in
        source)
            install_source_mode
            ;;
        tool)
            install_tool_mode
            ;;
        wheel)
            install_wheel_mode
            ;;
    esac
}

if [ "${GHOSTAP_INSTALL_LIBRARY_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

main "$@"
