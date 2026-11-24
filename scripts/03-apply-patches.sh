#!/bin/bash
# Apply the vLLM 0.30.0 runtime patches. Idempotent; originals are backed up.
# Usage: ./scripts/03-apply-patches.sh [--check-only]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1

PY="$(vllm_python)"
SP="$(vllm_site_packages)"
ROOT="$(vllm_root)"
check_vllm_version

QSA_OPS="$ROOT/models/qwen4_exp/nvidia/ops/qsa.py"
QSA_OWNER="$ROOT/models/qwen4_exp/nvidia/qsa.py"
QSA_CACHE="$ROOT/models/qwen4_exp/common/qsa_cache.py"
MODELOPT="$ROOT/model_executor/layers/quantization/modelopt.py"

if [[ "$CHECK_ONLY" != "1" ]]; then
    BK="$REPO_DIR/build/pre-qwen38-$(date +%Y%m%d%H%M%S)"
    mkdir -p "$BK"
    for f in "$ROOT/v1/worker/gpu/model_states/mamba_hybrid.py" \
             "$ROOT/v1/core/sched/scheduler.py" "$QSA_OPS" "$QSA_OWNER" \
             "$QSA_CACHE" "$MODELOPT"; do
        mkdir -p "$BK/$(dirname "${f#$ROOT/}")"
        cp -a "$f" "$BK/${f#$ROOT/}"
    done
    log "backups: $BK"
fi

log "02a FLA shared-memory gate"
"$PY" "$REPO_DIR/patches/02a-fla-shmem.py" --vllm-root "$ROOT" $([[ "$CHECK_ONLY" == "1" ]] && printf -- --check)

log "02b mamba prefix-cache block size"
if grep -q 'mamba_block_size or self.cache_config.block_size' "$ROOT/v1/worker/gpu/model_states/mamba_hybrid.py" &&
   grep -q 'self.block_size  # scheduler block size' "$ROOT/v1/core/sched/scheduler.py"; then
    log "already patched"
else
    [[ "$CHECK_ONLY" == "1" ]] && die "02b not applied"
    "$PY" "$REPO_DIR/patches/02b-mamba-block-size.py" "$SP"
fi

log "02c QSA fp8_e4m3 main KV cache"
if grep -q 'KV_QUANT_MODE' "$QSA_OPS"; then
    log "already patched"
else
    [[ "$CHECK_ONLY" == "1" ]] && die "02c not applied"
    "$PY" "$REPO_DIR/patches/02c-qsa-fp8-kv.py" "$SP"
fi

log "02g QSA fused-draft metadata"
if grep -q 'supports_draft_decode_metadata_update' "$QSA_CACHE"; then
    log "already patched"
else
    [[ "$CHECK_ONLY" == "1" ]] && die "02g not applied"
    VLLM_ROOT="$ROOT" "$PY" "$REPO_DIR/patches/02g-qsa-spec-cache.py"
fi

log "02h hybrid FP8 side-layer shim"
if [[ "$CHECK_ONLY" != "1" ]]; then
    cp "$REPO_DIR/patches/vllm_fp8_hybrid_modelopt.py" "$SP/vllm_fp8_hybrid_modelopt.py"
    if ! grep -q '_fp8_hybrid_apply' "$MODELOPT"; then
        printf '\nfrom vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >>"$MODELOPT"
    fi
    if ! grep -q '_fp8_hybrid_excluded(quant_config)' "$QSA_OWNER"; then
        sed -i 's/quant_config=model\.without_modelopt_fp4(quant_config)/quant_config=_fp8_hybrid_excluded(quant_config)/' "$QSA_OWNER"
        sed -i 's/^from \. import model$/from . import model\nfrom vllm_fp8_hybrid_modelopt import excluded_quant_config as _fp8_hybrid_excluded/' "$QSA_OWNER"
    fi
fi
grep -q '_fp8_hybrid_apply' "$MODELOPT" || die "hybrid shim hook missing in modelopt.py"
grep -q '_fp8_hybrid_excluded(quant_config)' "$QSA_OWNER" || die "hybrid shim hook missing in qsa.py"

log "syntax check"
"$PY" - <<PY
import ast
for p in ["$MODELOPT", "$QSA_OWNER", "$QSA_OPS", "$QSA_CACHE"]:
    ast.parse(open(p).read())
print("patch syntax OK")
PY

if [[ "$CHECK_ONLY" == "1" ]]; then
    log "all patches present"
else
    log "patches applied"
fi
