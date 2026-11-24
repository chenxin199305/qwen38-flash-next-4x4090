#!/bin/bash
# Health, model list and GPU occupancy.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

HOST="${BIND_HOST:-127.0.0.1}"
[[ "$HOST" == "0.0.0.0" ]] && HOST="127.0.0.1"
PORT="${PORT:-12345}"
BASE="http://$HOST:$PORT"

log "GET $BASE/health"
curl -fsS --max-time 5 "$BASE/health" && printf '\n'

API_KEY=""
if [[ -n "${API_KEY_FILE:-}" && -r "$API_KEY_FILE" ]]; then
    API_KEY="$(head -n1 "$API_KEY_FILE")"
elif [[ -n "${VLLM_API_KEY:-}" ]]; then
    API_KEY="$VLLM_API_KEY"
fi

log "GET $BASE/v1/models"
AUTH=()
[[ -n "$API_KEY" ]] && AUTH=(-H "Authorization: Bearer $API_KEY")
curl -fsS --max-time 5 "${AUTH[@]}" "$BASE/v1/models" | \
    "$(vllm_python 2>/dev/null || printf python3)" -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'

log "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
