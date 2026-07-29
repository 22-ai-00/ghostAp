#!/bin/bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)"
LOG_FILE="$PROJECT_DIR/logs.log"
PID_FILE="$PROJECT_DIR/.ghostap.pid"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

RESTART_GRACE_DELAY="${GHOSTAP_RESTART_GRACE_DELAY:-1}"
TERM_GRACE_DELAY="${GHOSTAP_TERM_GRACE_DELAY:-30}"
RESIDUAL_GRACE_DELAY="${GHOSTAP_RESIDUAL_GRACE_DELAY:-10}"
READINESS_TIMEOUT="${GHOSTAP_READINESS_TIMEOUT:-60}"
READINESS_POLL_INTERVAL="${GHOSTAP_READINESS_POLL_INTERVAL:-1}"
START_FAILURE_GRACE_DELAY="${GHOSTAP_START_FAILURE_GRACE_DELAY:-5}"
LOG_MODE="${GHOSTAP_LOG_MODE:-truncate}"
STARTED_PID=""
RESTART_REQUEST_SEQUENCE=0
PROJECT_LAUNCHCTL_ID=$(printf '%s' "$PROJECT_DIR" | cksum | awk '{print $1}')
LAUNCHCTL_LABEL="${GHOSTAP_LAUNCHCTL_LABEL:-com.ghostap.local.${PROJECT_LAUNCHCTL_ID}}"
CODEX_ACP_NPM_PACKAGE="${GHOSTAP_CODEX_ACP_NPM_PACKAGE:-@agentclientprotocol/codex-acp@1.1.2}"
PREPARE_CODEX_ACP="${GHOSTAP_PREPARE_CODEX_ACP:-1}"
TUI2ACP_NPM_PACKAGE="${GHOSTAP_TUI2ACP_NPM_PACKAGE:-tui2acp}"
PREPARE_TUI2ACP="${GHOSTAP_PREPARE_TUI2ACP:-1}"
SYNC_PYTHON_DEPENDENCIES="${GHOSTAP_SYNC_PYTHON_DEPENDENCIES:-1}"
PREPARE_EMPLOYEE_SANDBOX="${GHOSTAP_PREPARE_EMPLOYEE_SANDBOX:-1}"

cd "$PROJECT_DIR"

get_running_pids() {
    local pid command
    while read -r pid command; do
        [ -n "$pid" ] || continue
        if pid_is_ghostap_service "$pid" "$command"; then
            echo "$pid"
        fi
    done < <(ps -axo pid=,command= 2>/dev/null)
}

pid_belongs_to_project() {
    local pid="$1"
    local process_cwd=""
    case "$(uname -s)" in
        Linux)
            process_cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null) || return 1
            ;;
        Darwin)
            command -v lsof >/dev/null 2>&1 || return 1
            process_cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)
            ;;
        *)
            command -v pwdx >/dev/null 2>&1 || return 1
            process_cwd=$(pwdx "$pid" 2>/dev/null | sed 's/^[^:]*:[[:space:]]*//')
            ;;
    esac
    [ -n "$process_cwd" ] && [ "$process_cwd" = "$PROJECT_DIR" ]
}

pid_is_ghostap_service() {
    local pid="$1"
    local process_command="${2:-}"
    local command_kind=""
    if [ -z "$process_command" ]; then
        process_command=$(ps -p "$pid" -o command= 2>/dev/null) || return 1
    fi
    pid_belongs_to_project "$pid" || return 1
    case "$process_command" in
        "$PYTHON_BIN -m src.main"|".venv/bin/python -m src.main")
            command_kind="python"
            ;;
        "uv run python -m src.main"|*/uv\ run\ python\ -m\ src.main)
            command_kind="uv"
            ;;
        *) return 1 ;;
    esac
    if [ "$(uname -s)" = "Linux" ]; then
        local process_exe expected_exe
        process_exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || return 1
        if [ "$command_kind" = "python" ]; then
            expected_exe=$(readlink -f "$PYTHON_BIN" 2>/dev/null) || return 1
        else
            expected_exe=$(command -v uv 2>/dev/null) || return 1
            expected_exe=$(readlink -f "$expected_exe" 2>/dev/null) || return 1
        fi
        [ "$process_exe" = "$expected_exe" ] || return 1
    fi
    return 0
}

log_restart() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [RESTART] $*" >> "$LOG_FILE"
}

build_restart_python_command() {
    if [ -x "$PYTHON_BIN" ]; then
        RESTART_PYTHON_COMMAND=("$PYTHON_BIN")
    else
        RESTART_PYTHON_COMMAND=(uv run --project "$PROJECT_DIR" python)
    fi
}

verify_service_readiness() {
    local service_pid="$1"
    build_restart_python_command
    "${RESTART_PYTHON_COMMAND[@]}" -m src.utils.restart_gate ready \
        --project-dir "$PROJECT_DIR" \
        --service-pid "$service_pid"
}

publish_restart_generation() {
    local service_pid="${1:-${PID:-}}"
    [ -n "$service_pid" ] || return 1
    verify_service_readiness "$service_pid" >/dev/null
}

snapshot_restart_generation() {
    build_restart_python_command
    "${RESTART_PYTHON_COMMAND[@]}" -m src.utils.restart_gate snapshot \
        --project-dir "$PROJECT_DIR"
}

start_service_process() {
    local mode="${1:-truncate}"
    local detach_cmd=()
    if command -v setsid >/dev/null 2>&1; then
        detach_cmd=(setsid)
    elif command -v launchctl >/dev/null 2>&1; then
        launchctl remove "$LAUNCHCTL_LABEL" >/dev/null 2>&1 || true
        unset VIRTUAL_ENV
        if [ -x "$PYTHON_BIN" ]; then
            launchctl submit -l "$LAUNCHCTL_LABEL" -- /bin/bash -lc \
                'cd "$1" && exec "$2" -m src.main >>"$3" 2>&1' \
                bash "$PROJECT_DIR" "$PYTHON_BIN" "$LOG_FILE"
        else
            launchctl submit -l "$LAUNCHCTL_LABEL" -- /bin/bash -lc \
                'cd "$1" && exec uv run python -m src.main >>"$2" 2>&1' \
                bash "$PROJECT_DIR" "$LOG_FILE"
        fi
        STARTED_PID=""
        return
    fi

    unset VIRTUAL_ENV
    if [ -x "$PYTHON_BIN" ]; then
        if [ "$mode" = "append" ]; then
            nohup "${detach_cmd[@]}" "$PYTHON_BIN" -m src.main >> "$LOG_FILE" 2>&1 &
        else
            nohup "${detach_cmd[@]}" "$PYTHON_BIN" -m src.main > "$LOG_FILE" 2>&1 &
        fi
    else
        if [ "$mode" = "append" ]; then
            nohup "${detach_cmd[@]}" uv run python -m src.main >> "$LOG_FILE" 2>&1 &
        else
            nohup "${detach_cmd[@]}" uv run python -m src.main > "$LOG_FILE" 2>&1 &
        fi
    fi
    STARTED_PID=$!
    disown "$STARTED_PID" 2>/dev/null || true
}

wait_for_service_readiness() {
    local preferred_pid="${1:-}"
    local previous_pids="${2:-}"
    local deadline=$((SECONDS + READINESS_TIMEOUT))
    local candidates candidate generation previous

    while :; do
        if [ -n "$preferred_pid" ]; then
            candidates="$preferred_pid"
        else
            candidates=$(get_running_pids)
        fi
        for candidate in $candidates; do
            previous=0
            for prior_pid in $previous_pids; do
                if [ "$candidate" = "$prior_pid" ]; then
                    previous=1
                    break
                fi
            done
            [ "$previous" = "0" ] || continue
            if generation=$(verify_service_readiness "$candidate" 2>/dev/null) &&
                [[ "$generation" =~ ^[A-Za-z0-9_-]{20,64}$ ]]; then
                PID="$candidate"
                READY_GENERATION="$generation"
                RUNNING_PIDS=$(get_running_pids)
                [ -n "$RUNNING_PIDS" ] || RUNNING_PIDS="$candidate"
                return 0
            fi
        done

        if [ -n "$preferred_pid" ] && ! kill -0 "$preferred_pid" 2>/dev/null; then
            return 1
        fi
        if (( SECONDS >= deadline )); then
            return 1
        fi
        sleep "$READINESS_POLL_INTERVAL"
    done
}

service_command_label() {
    if [ -x "$PYTHON_BIN" ]; then
        echo "$PYTHON_BIN -m src.main"
    else
        echo "uv run python -m src.main"
    fi
}

venv_has_stale_entrypoint_shebang() {
    [ -d "$PROJECT_DIR/.venv/bin" ] || return 1
    local script first_line interpreter
    while IFS= read -r -d '' script; do
        IFS= read -r first_line < "$script" || continue
        case "$first_line" in
            '#!'*/.venv/bin/python*)
                interpreter="${first_line#\#!}"
                case "$interpreter" in
                    "$PROJECT_DIR"/.venv/bin/python*) ;;
                    *) return 0 ;;
                esac
                ;;
        esac
    done < <(find "$PROJECT_DIR/.venv/bin" -maxdepth 1 -type f -perm -u+x -print0)
    return 1
}

prepare_python_dependencies() {
    if [ "$SYNC_PYTHON_DEPENDENCIES" = "0" ]; then
        log_restart "python dependency sync skipped"
        return
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "❌ 未找到 uv，无法同步 Python 依赖"
        log_restart "python dependency sync failed missing uv"
        return 1
    fi

    echo "同步 GhostAP Python 依赖..."
    if venv_has_stale_entrypoint_shebang; then
        echo "检测到项目目录迁移，正在重建虚拟环境入口脚本..."
        if uv sync --group dev --reinstall >/dev/null 2>&1; then
            log_restart "python dependencies reinstalled stale entrypoint shebang"
            return
        fi
        echo "❌ Python 虚拟环境入口脚本修复失败"
        log_restart "python dependency reinstall failed stale entrypoint shebang"
        return 1
    elif uv sync --check --group dev >/dev/null 2>&1; then
        log_restart "python dependencies already synchronized"
        return
    fi
    if uv sync --group dev >/dev/null 2>&1; then
        log_restart "python dependencies synced"
    else
        echo "❌ Python 依赖同步失败"
        log_restart "python dependency sync failed"
        return 1
    fi
}

run_privileged() {
    if [ "$(id -u)" = "0" ]; then
        "$@"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        return 1
    fi
    if [ -t 0 ]; then
        sudo "$@"
    else
        sudo -n "$@"
    fi
}

install_linux_bubblewrap() {
    if command -v apt-get >/dev/null 2>&1; then
        run_privileged apt-get update >/dev/null 2>&1 && \
            run_privileged apt-get install -y bubblewrap >/dev/null 2>&1
    elif command -v dnf >/dev/null 2>&1; then
        run_privileged dnf install -y bubblewrap >/dev/null 2>&1
    elif command -v yum >/dev/null 2>&1; then
        run_privileged yum install -y bubblewrap >/dev/null 2>&1
    elif command -v zypper >/dev/null 2>&1; then
        run_privileged zypper --non-interactive install bubblewrap >/dev/null 2>&1
    elif command -v pacman >/dev/null 2>&1; then
        run_privileged pacman -S --needed --noconfirm bubblewrap >/dev/null 2>&1
    elif command -v apk >/dev/null 2>&1; then
        run_privileged apk add bubblewrap >/dev/null 2>&1
    else
        return 1
    fi
}

prepare_employee_sandbox_dependency() {
    if [ "$PREPARE_EMPLOYEE_SANDBOX" = "0" ]; then
        log_restart "employee sandbox preparation skipped"
        return
    fi

    case "$(uname -s)" in
        Linux)
            if [ ! -x /usr/bin/bwrap ]; then
                echo "安装员工 Channel 隔离依赖 bubblewrap..."
                install_linux_bubblewrap || true
            fi
            if [ -x /usr/bin/bwrap ]; then
                log_restart "employee sandbox ready mechanism=bubblewrap"
            else
                echo "⚠️  bubblewrap 自动安装失败，员工 Channel 将记录未验证降级"
                log_restart "employee sandbox unavailable mechanism=bubblewrap"
            fi
            ;;
        Darwin)
            if [ -x /usr/bin/sandbox-exec ]; then
                log_restart "employee sandbox ready mechanism=seatbelt"
            else
                echo "⚠️  macOS 系统 sandbox-exec 不可用，员工 Channel 将 fail-closed"
                log_restart "employee sandbox unavailable mechanism=seatbelt"
            fi
            ;;
        *)
            echo "⚠️  当前平台没有受支持的员工 Channel 文件系统沙箱"
            log_restart "employee sandbox unavailable mechanism=unsupported"
            ;;
    esac
}

codex_native_acp_available() {
    command -v codex >/dev/null 2>&1 || return 1
    codex acp serve --help 2>&1 | grep -Eiq "(acp serve|acp.*server)"
}

prepare_codex_acp_dependency() {
    if [ "$PREPARE_CODEX_ACP" = "0" ]; then
        log_restart "codex acp fallback preheat skipped"
        return
    fi
    if codex_native_acp_available; then
        log_restart "codex native acp serve available"
        return
    fi
    if ! command -v npx >/dev/null 2>&1; then
        echo "⚠️  未找到 npx，Codex ACP fallback 可能无法启动"
        log_restart "codex acp fallback missing npx"
        return
    fi

    echo "准备 Codex ACP fallback 依赖..."
    if npx --yes "$CODEX_ACP_NPM_PACKAGE" --version >/dev/null 2>&1; then
        log_restart "codex acp fallback ready package=$CODEX_ACP_NPM_PACKAGE"
    else
        echo "⚠️  Codex ACP fallback 依赖预热失败，后续 /codex 可能启动失败"
        log_restart "codex acp fallback preheat failed package=$CODEX_ACP_NPM_PACKAGE"
    fi
}

prepare_tui2acp_dependency() {
    if [ "$PREPARE_TUI2ACP" = "0" ]; then
        log_restart "tui2acp preheat skipped"
        return
    fi
    if command -v tui2acp >/dev/null 2>&1; then
        log_restart "tui2acp already available"
        return
    fi
    if ! command -v npm >/dev/null 2>&1; then
        echo "⚠️  未找到 npm，tui2acp 无法自动安装"
        log_restart "tui2acp missing npm"
        return
    fi

    echo "准备 tui2acp 依赖..."
    if npm install -g "$TUI2ACP_NPM_PACKAGE" >/dev/null 2>&1; then
        log_restart "tui2acp installed package=$TUI2ACP_NPM_PACKAGE"
        echo "✅ tui2acp 已安装"
    else
        echo "⚠️  tui2acp 安装失败，后续 /tui2acp 可能无法使用"
        log_restart "tui2acp install failed package=$TUI2ACP_NPM_PACKAGE"
    fi
}

wait_for_pid_exit() {
    local target_pid="$1"
    local timeout="$2"
    local deadline=$((SECONDS + timeout))

    while kill -0 "$target_pid" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            return 1
        fi
        sleep 1
    done
    return 0
}

terminate_service_pid() {
    local target_pid="$1"
    local grace_delay="$2"

    kill "$target_pid" 2>/dev/null || true
    if wait_for_pid_exit "$target_pid" "$grace_delay"; then
        return 0
    fi

    echo "进程 $target_pid 未在宽限期内退出，正在强制终止..."
    kill -9 "$target_pid" 2>/dev/null || true
    kill -9 -- -"$target_pid" 2>/dev/null || true
    return 2
}

cleanup_failed_start() {
    local target_pid="${1:-}"

    [ -n "$target_pid" ] || return 0
    if kill -0 "$target_pid" 2>/dev/null && pid_is_ghostap_service "$target_pid"; then
        terminate_service_pid "$target_pid" "$START_FAILURE_GRACE_DELAY" || true
    fi
    if [ -f "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null)" = "$target_pid" ]; then
        rm -f "$PID_FILE"
    fi
}

stop_service() {
    local forced=0
    local target_pid=""
    local residual_pids=""
    local p=""

    echo "正在停止 GhostAP 服务..."
    log_restart "stop begin"

    if [ -f "$PID_FILE" ]; then
        target_pid=$(cat "$PID_FILE")
        if kill -0 "$target_pid" 2>/dev/null &&
            pid_is_ghostap_service "$target_pid"; then
            if ! terminate_service_pid "$target_pid" "$TERM_GRACE_DELAY"; then
                forced=1
            fi
            echo "已停止进程 PID: $target_pid"
        elif kill -0 "$target_pid" 2>/dev/null; then
            echo "忽略不属于当前项目的 PID 文件记录: $target_pid"
            log_restart "stale pid file ignored pid=$target_pid"
        fi
        rm -f "$PID_FILE"
    fi
    if command -v launchctl >/dev/null 2>&1; then
        launchctl remove "$LAUNCHCTL_LABEL" >/dev/null 2>&1 || true
    fi

    residual_pids=$(get_running_pids)
    if [ -n "$residual_pids" ]; then
        echo "发现残留进程: $(echo "$residual_pids" | tr '\n' ' ')，正在清理..."
        for p in $residual_pids; do
            if pid_is_ghostap_service "$p" &&
                ! terminate_service_pid "$p" "$RESIDUAL_GRACE_DELAY"; then
                forced=1
            fi
        done
    fi

    if [ "$forced" = "1" ]; then
        echo "⚠️  服务已强制停止；本次停止为降级完成"
        log_restart "stop degraded forced termination"
        return 2
    fi

    echo "✅ 服务已停止"
    log_restart "stop done graceful"
    return 0
}

start_service() {
    local previous_running_pids=""
    echo "正在启动 GhostAP 服务..."
    local start_log_mode="$LOG_MODE"
    if [ "$start_log_mode" != "append" ]; then
        : > "$LOG_FILE"
        start_log_mode="append"
    fi
    prepare_python_dependencies || return 1
    prepare_employee_sandbox_dependency
    prepare_codex_acp_dependency
    prepare_tui2acp_dependency
    log_restart "start begin cmd=$(service_command_label)"
    previous_running_pids=$(get_running_pids)
    start_service_process "$start_log_mode"
    PID="$STARTED_PID"
    if [ -n "$PID" ]; then
        echo "$PID" > "$PID_FILE"
    fi

    if ! wait_for_service_readiness "$PID" "$previous_running_pids"; then
        echo "❌ 启动失败：服务未在 ${READINESS_TIMEOUT}s 内通过 readiness 检查"
        log_restart "start failed readiness pid=$PID"
        cleanup_failed_start "$PID"
        return 1
    fi
    if [ -z "$PID" ]; then
        PID=$(echo "$RUNNING_PIDS" | awk 'NR==1 {print $1}')
    fi
    echo "$PID" > "$PID_FILE"
    if ! publish_restart_generation "$PID"; then
        echo "❌ 服务 readiness 已通过，但无法确认安全重启 generation"
        log_restart "start failed generation publish pid=$PID"
        cleanup_failed_start "$PID"
        return 1
    fi
    echo "✅ GhostAP 服务已启动且 readiness 已通过"
    echo "   进程: $RUNNING_PIDS"
    echo "   启动命令: $(service_command_label)"
    echo "   日志: $LOG_FILE"
    log_restart "start ready pid=$PID running=$RUNNING_PIDS generation=${READY_GENERATION:-verified}"
    return 0
}

show_status() {
    PIDS=$(get_running_pids)
    if [ -n "$PIDS" ]; then
        echo "✅ GhostAP 正在运行"
        echo "   进程列表:"
        for PID in $PIDS; do
            ps -p "$PID" -o pid=,command=
        done
    else
        echo "❌ GhostAP 未运行"
    fi
}

remote_restart() {
    echo "🔄 触发远程重启..."

    build_restart_python_command
    local expected_generation=""
    if ! expected_generation="$(snapshot_restart_generation)"; then
        echo "❌ 安全重启预检失败，未启动重启 worker"
        log_restart "remote worker preflight failed"
        return 1
    fi
    if [[ ! "$expected_generation" =~ ^[A-Za-z0-9_-]{20,64}$ ]]; then
        echo "❌ 安全重启预检返回了无效 generation"
        log_restart "remote worker preflight invalid generation"
        return 1
    fi
    local worker_command=(
        "${RESTART_PYTHON_COMMAND[@]}"
        -m src.utils.restart_gate worker
        --project-dir "$PROJECT_DIR"
        --expected-generation "$expected_generation"
        --restart-script "$PROJECT_DIR/restart.sh"
        --log-file "$LOG_FILE"
        --delay "$RESTART_GRACE_DELAY"
    )
    local worker_pid=""

    case "$(uname -s)" in
        Linux)
            if ! command -v setsid >/dev/null 2>&1; then
                echo "❌ 当前 Linux 缺少 setsid，拒绝启动不安全的远程重启"
                log_restart "remote worker rejected missing setsid"
                return 1
            fi
            setsid "${worker_command[@]}" </dev/null >/dev/null 2>&1 &
            worker_pid=$!
            ;;
        Darwin)
            if ! command -v launchctl >/dev/null 2>&1; then
                echo "❌ 当前 macOS 缺少 launchctl，拒绝启动不安全的远程重启"
                log_restart "remote worker rejected missing launchctl"
                return 1
            fi
            RESTART_REQUEST_SEQUENCE=$((RESTART_REQUEST_SEQUENCE + 1))
            local request_label
            request_label="${LAUNCHCTL_LABEL}.restart.$$.$RESTART_REQUEST_SEQUENCE.$RANDOM"
            worker_command=(
                "${RESTART_PYTHON_COMMAND[@]}"
                -m src.utils.restart_gate launch-wrapper
                --project-dir "$PROJECT_DIR"
                --expected-generation "$expected_generation"
                --restart-script "$PROJECT_DIR/restart.sh"
                --log-file "$LOG_FILE"
                --delay "$RESTART_GRACE_DELAY"
                --launchd-label "$request_label"
            )
            if ! launchctl submit -l "$request_label" -- \
                "${worker_command[@]}" >/dev/null 2>&1; then
                echo "❌ launchctl 无法启动安全重启 worker"
                log_restart "remote worker launch failed label=$request_label"
                return 1
            fi
            ;;
        *)
            echo "❌ 当前平台不支持安全远程重启"
            log_restart "remote worker rejected unsupported platform=$(uname -s)"
            return 1
            ;;
    esac
    if [ -n "$worker_pid" ]; then
        disown "$worker_pid" 2>/dev/null || true
    fi
    
    echo "✅ 远程重启已触发"
    echo "   服务将在当前任务安全结束后重新启动"
    echo "   查看日志: tail -f $LOG_FILE"
}

if [ "${GHOSTAP_RESTART_LIBRARY_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

case "${1:-restart}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_status=0
        stop_service || stop_status=$?
        LOG_MODE="${GHOSTAP_LOG_MODE:-append}"
        if ! start_service; then
            exit 1
        fi
        exit "$stop_status"
        ;;
    remote-restart|rr)
        remote_restart
        ;;
    status)
        show_status
        ;;
    *)
        echo "用法: $0 {start|stop|restart|remote-restart|status}"
        echo "  start          - 启动服务"
        echo "  stop           - 停止服务"
        echo "  restart        - 本地重启（停止后立即启动）"
        echo "  remote-restart - 远程重启（适用于通过机器人执行）"
        echo "  status         - 查看服务状态"
        exit 1
        ;;
esac
