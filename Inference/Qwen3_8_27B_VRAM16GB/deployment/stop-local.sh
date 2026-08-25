#!/usr/bin/env bash
# Stop the locally started AccessPoint (start-local.sh pid files).
set -euo pipefail
RUNDIR="${XDG_RUNTIME_DIR:-/tmp}/hga-qwen38"
stop_pid() {
  local name="$1" file="$2"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "stopping $name pid=$pid" >&2
      kill "$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}
stop_pid gateway "$RUNDIR/gateway.pid"
stop_pid server "$RUNDIR/server.pid"
# run-api.sh execs llama-server; catch a leftover if the pid file was stale.
pkill -f "llama-server.*qwen3.8-27b-hga" 2>/dev/null || true
echo "AccessPoint stopped"
