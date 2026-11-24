#!/bin/bash
# Collect environment and recent error lines for a bug report.
# Redacts nothing automatically: review the output before pasting it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

LOG_DIR="${LOG_DIR:-$HOME/.local/state/qwen38-flash-next}"

section() { printf '\n== %s ==\n' "$*"; }

section "host"
uname -a
nproc
free -g | sed -n '1,2p'
df -h "$REPO_DIR" | tail -n1

section "gpu"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true

section "cuda"
"${CUDA_HOME:-/usr/local/cuda-12.8}/bin/nvcc" --version 2>/dev/null | tail -n2 || echo "nvcc missing"

section "python"
if [[ -x "${VENV:-$HOME/vllm-venv}/bin/python" ]]; then
    "$(vllm_python)" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
try:
    import vllm.vllm_flash_attn as fa
    print("FA2_AVAILABLE", fa.FA2_AVAILABLE,
          "varlen_fwd", hasattr(torch.ops._vllm_fa2_C, "varlen_fwd"))
except Exception as e:
    print("flash_attn import failed:", e)
PY
else
    echo "venv missing"
fi

section "service"
if systemctl list-unit-files vllm-qwen38-tp4.service >/dev/null 2>&1; then
    systemctl --user --no-pager status vllm-qwen38-tp4 2>/dev/null | sed -n '1,15p' || true
fi
[[ -f "${LOG_DIR}/vllm.pid" ]] && echo "pid file: $(cat "${LOG_DIR}/vllm.pid")"
for ctl in /etc/dsw/runtime/bin/supervisorctl; do
    [[ -x "$ctl" && -f /etc/dsw/sys_configs/supervisor.conf ]] && \
        "$ctl" -c /etc/dsw/sys_configs/supervisor.conf status vllm-tp4 2>/dev/null || true
done

section "recent errors"
LOGFILE="${LOG_DIR}/vllm.log"
[[ -f "$LOGFILE" ]] || LOGFILE=/var/log/vllm-tp4.log
if [[ -f "$LOGFILE" ]]; then
    tail -n 500 "$LOGFILE" | grep -nE 'Traceback|Error|error|FAILED|out of memory|invalid resource' | tail -n 40 || true
else
    echo "no log found"
fi
