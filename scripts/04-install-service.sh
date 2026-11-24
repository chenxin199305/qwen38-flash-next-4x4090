#!/bin/bash
# Install a supervisor (DSW/PAI) or systemd user service.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

VENV="${VENV:-$HOME/vllm-venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
LOG_DIR="${LOG_DIR:-$HOME/.local/state/qwen38-flash-next}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vllm.log}"
mkdir -p "$LOG_DIR"

render() {
    sed -e "s|@REPO_DIR@|$REPO_DIR|g" \
        -e "s|@VENV@|$VENV|g" \
        -e "s|@CUDA_HOME@|$CUDA_HOME|g" \
        -e "s|@HOME_DIR@|$HOME|g" \
        -e "s|@LOG_FILE@|$LOG_FILE|g" "$1"
}

SUPERVISOR_CONF="${SUPERVISOR_CONF:-/etc/dsw/sys_configs/supervisor.conf}"
SUPERVISORCTL="${SUPERVISORCTL:-/etc/dsw/runtime/bin/supervisorctl}"

if [[ -f "$SUPERVISOR_CONF" && -x "$SUPERVISORCTL" ]]; then
    log "supervisor detected"
    DEST_DIR="$(dirname "$SUPERVISOR_CONF")/supervisor/conf.d"
    mkdir -p "$DEST_DIR"
    render "$REPO_DIR/services/supervisor-vllm-tp4.conf.in" >"$DEST_DIR/supervisor-vllm-tp4.conf"
    "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" reread
    "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" update
    "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" status vllm-tp4 || true
    log "installed $DEST_DIR/supervisor-vllm-tp4.conf"
    exit 0
fi

if command -v systemctl >/dev/null 2>&1 && [[ "$(ps -p 1 -o comm= 2>/dev/null || true)" == "systemd" ]]; then
    UNIT_DIR="$HOME/.config/systemd/user"
    UNIT="$UNIT_DIR/vllm-qwen38-tp4.service"
    mkdir -p "$UNIT_DIR"
    render "$REPO_DIR/services/vllm-qwen38-tp4.service.in" >"$UNIT"
    systemctl --user daemon-reload
    systemctl --user enable --now vllm-qwen38-tp4
    systemctl --user --no-pager status vllm-qwen38-tp4 || true
    log "installed $UNIT"
    exit 0
fi

warn "no supervisor or systemd found; use ./start.sh and ./stop.sh"
