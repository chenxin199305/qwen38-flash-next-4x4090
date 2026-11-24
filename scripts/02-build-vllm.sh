#!/bin/bash
# Build vLLM 0.30.0 from source for driver 570 / CUDA 12.8.
#
# The critical step is the FA2 extension: the wheel must contain
# vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so. Without it the vision-encoder
# profile run at startup dies with
#   AttributeError: '_OpNamespace' '_vllm_fa2_C' object has no attribute 'varlen_fwd'
# This script therefore builds the upstream vllm-flash-attn subproject from a
# local checkout and links it with LIBRARY_PATH pointing at CUDA 12.8.
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
DIST="$REPO_DIR/dist"
DEPS="$REPO_DIR/build/deps"
MAX_JOBS="${MAX_JOBS:-32}"
FORCE="${FORCE:-0}"

ensure_torch_cu126() {
    if ! "$VENV/bin/python" -c 'import torch,sys; sys.exit(0 if torch.version.cuda == "12.6" else 1)'; then
        log "restoring torch cu126 (the wheel install changed it)"
        UV_INDEX_URL="https://download.pytorch.org/whl/cu126" uv pip install \
            --python "$VENV/bin/python" --force-reinstall \
            "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"
    fi
}

install_wheel() {
    local whl="$1"
    need_file "$whl"
    log "install $whl"
    uv pip install --python "$VENV/bin/python" "$whl"
    ensure_torch_cu126
    "$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
    if ! "$VENV/bin/python" - <<'PY'
import torch, sys
try:
    import vllm.vllm_flash_attn as fa
    ok = fa.FA2_AVAILABLE and hasattr(torch.ops._vllm_fa2_C, "varlen_fwd")
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
    then
        warn "wheel has no working FA2 extension; building it separately"
        "$SCRIPT_DIR/02b-build-fa2.sh"
    fi
}

if [[ -n "${VLLM_WHEEL:-}" && -f "${VLLM_WHEEL}" ]]; then
    install_wheel "$VLLM_WHEEL"
    exit 0
fi

if [[ "$FORCE" != "1" && -x "$VENV/bin/vllm" ]] && \
   "$VENV/bin/python" -c "import vllm; assert vllm.__version__ == '$VLLM_VERSION'" 2>/dev/null; then
    log "vLLM $VLLM_VERSION already installed in $VENV (FORCE=1 to rebuild)"
    "$VENV/bin/python" - <<'PY' || "$SCRIPT_DIR/02b-build-fa2.sh"
import torch, sys
import vllm.vllm_flash_attn as fa
sys.exit(0 if fa.FA2_AVAILABLE and hasattr(torch.ops._vllm_fa2_C, "varlen_fwd") else 1)
PY
    exit 0
fi

log "source checkouts"
for cmd in git cmake ninja; do require_cmd "$cmd"; done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "nvcc not found under $CUDA_HOME"
if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
require_cmd cargo

if [[ ! -d "$VLLM_SRC/.git" ]]; then
    git clone --depth 1 --branch "v$VLLM_VERSION" \
        https://github.com/vllm-project/vllm.git "$VLLM_SRC" 2>/dev/null || {
        git clone --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_SRC"
        git -C "$VLLM_SRC" fetch --depth 1 origin "v$VLLM_VERSION"
        git -C "$VLLM_SRC" checkout --detach FETCH_HEAD
    }
fi

if [[ ! -d "$FA_SRC/.git" ]]; then
    git clone https://github.com/vllm-project/flash-attention.git "$FA_SRC"
fi
git -C "$FA_SRC" checkout --detach "$FA_COMMIT"
git -C "$FA_SRC" submodule update --init csrc/cutlass

mkdir -p "$DEPS"
clone_dep() {
    local repo="$1" ref="$2" dest="$3"
    [[ -d "$dest" ]] && return 0
    git clone --depth 1 --branch "$ref" "https://github.com/$repo.git" "$dest"
}
clone_dep NVIDIA/cutlass v4.7.1 "$DEPS/cutlass-4.7.1"
clone_dep triton-lang/triton v3.5.1 "$DEPS/triton-3.5.1"

# vLLM gates vllm-flash-attn behind an external-project include. Put it back
# after the triton_kernels include so the FA subproject wins macro collisions.
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

log "build wheel (MAX_JOBS=$MAX_JOBS, archs=$(cuda_arch_list))"
mkdir -p "$DIST"
RUST_BIN="$(dirname "$(command -v cargo)")"
(
    cd "$VLLM_SRC"
    export VLLM_CUTLASS_SRC_DIR="$DEPS/cutlass-4.7.1" \
        TRITON_KERNELS_SRC_DIR="$DEPS/triton-3.5.1/python/triton_kernels/triton_kernels" \
        VLLM_FLASH_ATTN_SRC_DIR="$FA_SRC" \
        CUDA_HOME="$CUDA_HOME" \
        LIBRARY_PATH="$CUDA_HOME/lib64" \
        PATH="$RUST_BIN:$VENV/bin:$CUDA_HOME/bin:$PATH" \
        TORCH_CUDA_ARCH_LIST="$(cuda_arch_list)" \
        MAX_JOBS="$MAX_JOBS" \
        VLLM_TARGET_DEVICE=cuda
    "$VENV/bin/python" setup.py bdist_wheel --dist-dir "$DIST"
)

WHL="$(find "$DIST" -maxdepth 1 -name "vllm-${VLLM_VERSION}*.whl" -printf '%T@ %p\n' | sort -n | tail -n1 | cut -d' ' -f2-)"
[[ -n "$WHL" ]] || die "wheel not produced"
install_wheel "$WHL"
log "done"
