#!/bin/bash
# Build and install only the FA2 extension into an existing vLLM venv.
# Use this to repair a wheel that was built before the vllm-flash-attn
# external project was wired in. Idempotent.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

VENV="${VENV:-$HOME/vllm-venv}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
VLLM_SRC="${VLLM_SRC:-$HOME/src/vllm-$VLLM_VERSION}"
FA_SRC="${FA_SRC:-$HOME/src/vllm-flash-attn}"
FA_COMMIT="${FA_COMMIT:-506341a143fcabd4bb79052a7605ada727d6b3f5}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
JOBS="${JOBS:-24}"
BUILD="$VLLM_SRC/build-fa2"
TMP="$REPO_DIR/build/fa2-install"
SP="$(vllm_site_packages)"

if "$(vllm_python)" - <<'PY' 2>/dev/null
import torch, sys
import vllm.vllm_flash_attn as fa
sys.exit(0 if fa.FA2_AVAILABLE and hasattr(torch.ops._vllm_fa2_C, "varlen_fwd") else 1)
PY
then
    log "FA2 already installed"
    exit 0
fi

need_dir "$VLLM_SRC"
for cmd in git cmake ninja; do require_cmd "$cmd"; done

if [[ ! -d "$FA_SRC/.git" ]]; then
    git clone https://github.com/vllm-project/flash-attention.git "$FA_SRC"
fi
git -C "$FA_SRC" checkout --detach "$FA_COMMIT"
git -C "$FA_SRC" submodule update --init csrc/cutlass

if ! grep -q 'external_projects/vllm_flash_attn.cmake' "$VLLM_SRC/CMakeLists.txt"; then
    python3 - "$VLLM_SRC/CMakeLists.txt" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "    include(cmake/external_projects/triton_kernels.cmake)\nendif()"
add = anchor + "\n\nif(VLLM_GPU_LANG STREQUAL \"CUDA\")\n    include(cmake/external_projects/vllm_flash_attn.cmake)\nendif()"
assert anchor in s, "triton_kernels include anchor not found"
open(p, "w").write(s.replace(anchor, add, 1))
PYEOF
fi

export CUDA_HOME="$CUDA_HOME" \
    TORCH_CUDA_ARCH_LIST="$(cuda_arch_list)" \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_FLASH_ATTN_SRC_DIR="$FA_SRC" \
    LIBRARY_PATH="$CUDA_HOME/lib64" \
    PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"

log "configure"
cmake -S "$VLLM_SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
log "build FA2 (JOBS=$JOBS)"
ninja -C "$BUILD" -j "$JOBS" _vllm_fa2_C

log "install into $SP"
rm -rf "$TMP"
cmake --install "$BUILD" --component _vllm_fa2_C --prefix "$TMP"
cp -a "$TMP/vllm/vllm_flash_attn/." "$SP/vllm/vllm_flash_attn/"
for f in layers/__init__.py ops/__init__.py ops/triton/__init__.py; do
    [[ -f "$FA_SRC/vllm_flash_attn/$f" ]] && cp "$FA_SRC/vllm_flash_attn/$f" "$SP/vllm/vllm_flash_attn/$f"
done

"$(vllm_python)" - <<'PY'
import torch
import vllm.vllm_flash_attn as fa
assert fa.FA2_AVAILABLE, "FA2 import failed"
assert hasattr(torch.ops._vllm_fa2_C, "varlen_fwd"), "varlen_fwd not registered"
print("FA2 OK:", fa.__file__)
PY
log "done"
