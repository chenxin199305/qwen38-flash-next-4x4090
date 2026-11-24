#!/bin/bash
# Minimal end-to-end chat request.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

HOST="${BIND_HOST:-127.0.0.1}"
[[ "$HOST" == "0.0.0.0" ]] && HOST="127.0.0.1"
PORT="${PORT:-12345}"
MODEL="${SERVED_MODEL_NAME:-qwen38-flash-next-nvfp4}"

API_KEY=""
if [[ -n "${API_KEY_FILE:-}" && -r "$API_KEY_FILE" ]]; then
    API_KEY="$(head -n1 "$API_KEY_FILE")"
elif [[ -n "${VLLM_API_KEY:-}" ]]; then
    API_KEY="$VLLM_API_KEY"
fi

AUTH=()
[[ -n "$API_KEY" ]] && AUTH=(-H "Authorization: Bearer $API_KEY")

log "chat completion"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
curl -fsS --max-time 300 "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    "${AUTH[@]}" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2等于几？只回答数字。\"}],\"max_tokens\":64,\"temperature\":0}" \
    >"$TMP"

"$(vllm_python 2>/dev/null || printf python3)" - "$TMP" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
msg = d["choices"][0]["message"]
text = (msg.get("content") or "") + (msg.get("reasoning_content")
                                     or msg.get("reasoning") or "")
print("content:", repr(msg.get("content")))
print("usage:", d.get("usage"))
assert text.strip(), "empty response"
assert "4" in text, f"unexpected answer: {text!r}"
print("smoke OK")
PY
