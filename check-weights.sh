#!/bin/bash
# Verify the base checkpoint and the FP8-hybrid overlay without loading the GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/lib/common.sh"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}"
OVERLAY="${FP8HYBRID_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4-fp8hybrid}"

need_file "$MODEL_DIR/config.json"
need_file "$MODEL_DIR/model.safetensors.index.json"
log "base checkpoint: $MODEL_DIR"
python3 - "$MODEL_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
idx = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not (root / s).is_file()]
print(f"{len(idx['weight_map'])} tensors, {len(shards)} shards, missing={len(missing)}")
if missing:
    raise SystemExit(f"missing shards: {missing[:5]}")
PY

if [[ ! -e "$OVERLAY" ]]; then
    warn "overlay not found: $OVERLAY"
    exit 0
fi

need_file "$OVERLAY/model.safetensors.index.json"
log "FP8-hybrid overlay: $OVERLAY"
python3 - "$OVERLAY" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
idx = json.loads((root / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
scales = [n for n in wm if n.endswith(".weight_scale_inv")]
print(f"{len(scales)} weight_scale_inv tensors")
if scales and len(scales) != 156:
    print(f"WARNING: expected 156 for the TP4 layout, found {len(scales)}")
broken = [p.name for p in root.iterdir() if p.is_symlink() and not p.exists()]
print(f"broken symlinks: {len(broken)}")
if broken:
    raise SystemExit(f"broken: {broken[:5]}")
missing = sorted({s for s in wm.values() if not (root / s).exists()})
if missing:
    raise SystemExit(f"missing overlay shards: {missing[:5]}")
PY

MANIFEST="$OVERLAY/fp8hybrid-manifest.json"
if [[ -f "$MANIFEST" ]]; then
    python3 - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("manifest:", {k: m.get(k) for k in
      ("target_tensors", "target_shards", "worst_rel_err", "block", "dtype")})
PY
else
    warn "no fp8hybrid-manifest.json (older in-place converter or manual build)"
fi
log "weights OK"
