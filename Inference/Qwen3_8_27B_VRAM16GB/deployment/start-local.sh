#!/usr/bin/env bash
# Start llama-server + the OpenAI profile gateway on this 16 GB GPU host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"
ENV_FILE="${HGA_API_ENV:-$HOME/.config/hga-qwen38/api.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
: "${HGA_API_KEY:?set HGA_API_KEY or create $ENV_FILE}"

export HGA_BACKEND_PORT="${HGA_BACKEND_PORT:-8081}"
export HGA_API_HOST="${HGA_API_HOST:-127.0.0.1}"
export HGA_API_PORT="${HGA_API_PORT:-8080}"
export HGA_BACKEND_URL="${HGA_BACKEND_URL:-http://127.0.0.1:${HGA_BACKEND_PORT}}"
RUNDIR="${XDG_RUNTIME_DIR:-/tmp}/hga-qwen38"
mkdir -p "$RUNDIR" "$HOME/.config/hga-qwen38"

if [[ -f "$RUNDIR/server.pid" ]] && kill -0 "$(cat "$RUNDIR/server.pid")" 2>/dev/null; then
  echo "llama-server already running pid=$(cat "$RUNDIR/server.pid")" >&2
else
  echo "==> starting llama-server on 127.0.0.1:${HGA_BACKEND_PORT}" >&2
  nohup "$ROOT/deployment/run-api.sh" >"$RUNDIR/server.log" 2>&1 &
  echo $! >"$RUNDIR/server.pid"
fi

python3 - "$HGA_BACKEND_URL" "$HGA_API_KEY" <<'PY'
import os, sys, time, urllib.request
url, key = sys.argv[1], sys.argv[2]
deadline = time.time() + 180
last = None
while time.time() < deadline:
    try:
        req = urllib.request.Request(url + "/health", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=5) as response:
            print(f"backend healthy HTTP {response.status}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last = exc
        time.sleep(2)
raise SystemExit(f"backend did not become healthy: {last}")
PY

if [[ -f "$RUNDIR/gateway.pid" ]] && kill -0 "$(cat "$RUNDIR/gateway.pid")" 2>/dev/null; then
  echo "gateway already running pid=$(cat "$RUNDIR/gateway.pid")" >&2
else
  echo "==> starting gateway on ${HGA_API_HOST}:${HGA_API_PORT}" >&2
  nohup python3 "$ROOT/deployment/api_gateway.py" \
    --host "$HGA_API_HOST" --port "$HGA_API_PORT" \
    >"$RUNDIR/gateway.log" 2>&1 &
  echo $! >"$RUNDIR/gateway.pid"
fi
sleep 1
echo "AccessPoint http://${HGA_API_HOST}:${HGA_API_PORT}/v1"
echo "logs: $RUNDIR/server.log  $RUNDIR/gateway.log"
