#!/bin/bash
# Preflight: hardware, driver, toolchain and model sanity.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

TP_SIZE="${TP_SIZE:-4}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}"

log "GPU check"
require_cmd nvidia-smi
COUNT="$(gpu_count)"
[[ "$COUNT" -ge "$TP_SIZE" ]] || die "found $COUNT GPUs, TP_SIZE=$TP_SIZE"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
LOW_MEM="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F', ' '$2 < 44000 {print $1" ("$2" MiB)"}')"
if [[ -n "$LOW_MEM" ]]; then
    warn "GPU(s) with < 44 GiB: $LOW_MEM; the model needs ~24 GiB weights/card plus KV"
fi
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
DRIVER_MAJOR="${DRIVER%%.*}"
if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ && "$DRIVER_MAJOR" -lt 570 ]]; then
    warn "driver $DRIVER is older than 570; CUDA 12.8 runtime may not load"
fi

log "host memory / disk"
RAM_GIB="$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))"
printf 'RAM: %s GiB\n' "$RAM_GIB"
[[ "$RAM_GIB" -ge 120 ]] || warn "128 GiB is the minimum; the PLE table is ~51 GiB in host RAM, 256 GiB recommended"
CHECK_DIR="$MODEL_DIR"
while [[ ! -e "$CHECK_DIR" && "$CHECK_DIR" != "/" ]]; do CHECK_DIR="$(dirname "$CHECK_DIR")"; done
FREE_GIB="$(( $(df -BG --output=avail "$CHECK_DIR" | tail -n1 | tr -dc '0-9') ))"
printf 'free disk under %s: %s GiB\n' "$CHECK_DIR" "$FREE_GIB"
[[ "$FREE_GIB" -ge 200 ]] || warn "~200 GiB free is recommended (126 GiB base + 28 GiB overlay + wheel/build)"

log "toolchain"
for cmd in python3 git uv cmake; do require_cmd "$cmd"; done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "nvcc not found under $CUDA_HOME (install the full CUDA 12.8 toolkit)"
"$CUDA_HOME/bin/nvcc" --version | tail -n 2
for lib in libcudadevrt.a libcudart_static.a; do
    [[ -f "$CUDA_HOME/lib64/$lib" ]] || die "missing $CUDA_HOME/lib64/$lib (install the full CUDA 12.8 toolkit)"
done
if ! command -v ninja >/dev/null 2>&1 && [[ ! -x "${VENV:-$HOME/vllm-venv}/bin/ninja" ]]; then
    warn "ninja not found; scripts/01-setup-venv.sh installs it"
fi
if ! command -v cargo >/dev/null 2>&1 && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
    warn "cargo/rustc not found; vLLM 0.30 source build needs a Rust toolchain (rustup)"
fi

log "port"
PORT="${PORT:-12345}"
port_in_use "$PORT" && die "port $PORT is already in use"

log "model"
if [[ -f "$MODEL_DIR/config.json" ]]; then
    "$(vllm_python 2>/dev/null || printf python3)" - "$MODEL_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cfg = json.loads((root / "config.json").read_text())
t = cfg.get("text_config", cfg)
print(f"layers={t.get('num_hidden_layers')} hidden={t.get('hidden_size')} "
      f"experts={t.get('num_experts')} moe_ffn={t.get('moe_intermediate_size')}")
idx = root / "model.safetensors.index.json"
print("index:", "ok" if idx.is_file() else "MISSING")
PY
else
    warn "base checkpoint not found at $MODEL_DIR (download it before ./install.sh)"
fi

log "preflight ok"
