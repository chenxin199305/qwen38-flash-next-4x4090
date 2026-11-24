#!/bin/bash
# One-shot installer:
#   preflight -> venv -> vLLM source build (with FA2) -> FP8 hybrid overlay
#   -> runtime patches -> service.
# Every step is idempotent.
#
# Usage:
#   ./install.sh [--skip-preflight] [--skip-venv] [--skip-build]
#                [--skip-fp8hybrid] [--skip-patches] [--skip-service]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/lib/common.sh"

SKIP_PREFLIGHT=0 SKIP_VENV=0 SKIP_BUILD=0 SKIP_FP8HYBRID=0 SKIP_PATCHES=0 SKIP_SERVICE=0
for arg in "$@"; do
    case "$arg" in
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --skip-venv) SKIP_VENV=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        --skip-fp8hybrid) SKIP_FP8HYBRID=1 ;;
        --skip-patches) SKIP_PATCHES=1 ;;
        --skip-service) SKIP_SERVICE=1 ;;
        *) die "unknown flag: $arg" ;;
    esac
done

[[ "$SKIP_PREFLIGHT" == "1" ]] || "$SCRIPT_DIR/scripts/00-preflight.sh"
[[ "$SKIP_VENV" == "1" ]] || "$SCRIPT_DIR/scripts/01-setup-venv.sh"
[[ "$SKIP_BUILD" == "1" ]] || "$SCRIPT_DIR/scripts/02-build-vllm.sh"

if [[ "$SKIP_FP8HYBRID" == "1" ]]; then
    log "skip FP8-hybrid overlay"
elif [[ -f "${FP8HYBRID_DIR:-}/fp8hybrid-manifest.json" ]]; then
    log "FP8-hybrid overlay already exists; verifying"
    "$(vllm_python)" "$SCRIPT_DIR/scripts/02a-build-fp8hybrid.py" --check
elif [[ -f "${FP8HYBRID_DIR:-}/model.safetensors.index.json" ]]; then
    warn "FP8-hybrid overlay exists without a manifest; skipping rebuild"
    warn "delete ${FP8HYBRID_DIR:-<unset>} and re-run for a clean 156-tensor overlay"
else
    "$(vllm_python)" "$SCRIPT_DIR/scripts/02a-build-fp8hybrid.py"
fi

[[ "$SKIP_PATCHES" == "1" ]] || "$SCRIPT_DIR/scripts/03-apply-patches.sh"
[[ "$SKIP_SERVICE" == "1" ]] || "$SCRIPT_DIR/scripts/04-install-service.sh"

cat <<EOF

==> install complete

Start:
  ./start.sh --wait
  ./scripts/healthcheck.sh
  ./scripts/smoke.sh

Logs:
  ${LOG_DIR:-$HOME/.local/state/qwen38-flash-next}/vllm.log

If neither supervisor nor systemd is available, ./start.sh and ./stop.sh are
the supported path.
EOF
