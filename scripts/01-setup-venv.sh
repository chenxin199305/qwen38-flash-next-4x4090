#!/bin/bash
# Create the venv, install the CUDA 12.6 torch stack and the build toolchain.
# vLLM itself is installed by scripts/02-build-vllm.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

VENV="${VENV:-$HOME/vllm-venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
PYPI="${PIP_INDEX_URL:-https://pypi.org/simple}"

log "requirements"
for cmd in uv python3 git; do require_cmd "$cmd"; done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "nvcc not found under $CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | tail -n 2

if [[ ! -x "$VENV/bin/python" ]]; then
    log "create venv: $VENV"
    uv venv "$VENV" --python "$(command -v python3)"
else
    log "venv exists: $VENV"
fi

log "torch 2.13.0 + cu126"
UV_INDEX_URL="$TORCH_INDEX" uv pip install --python "$VENV/bin/python" \
    "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"
UV_INDEX_URL="$TORCH_INDEX" uv pip install --python "$VENV/bin/python" \
    "torchcodec==0.17.0"

log "build and runtime helpers"
uv pip install --python "$VENV/bin/python" \
    "pip" "cmake>=3.26.1" "ninja" "packaging>=24.2" \
    "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" "setuptools-rust>=1.9.0" \
    "wheel" "jinja2>=3.1.6" "regex" "build" "protobuf>=6.33.5,<8.0" \
    --index-url "$PYPI"

"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see CUDA"
print("torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.version.cuda == "12.6", "expected the cu126 wheel"
PY
log "venv ready: $VENV"
