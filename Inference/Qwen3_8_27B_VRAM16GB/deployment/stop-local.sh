#!/usr/bin/env bash
# Stop the AccessPoint: systemd user units if reachable, else pid files.
#
# Desktop/IDE shells often keep DBUS_SESSION_BUS_ADDRESS pointed at a GNOME
# session bus under /tmp. systemctl --user then fails with:
#   Process org.freedesktop.systemd1 exited with status 1
# Address the persistent per-user manager at $XDG_RUNTIME_DIR/bus instead.
set -euo pipefail

runtime="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [[ -S "${runtime}/bus" ]]; then
  export XDG_RUNTIME_DIR="$runtime"
  export DBUS_SESSION_BUS_ADDRESS="unix:path=${runtime}/bus"
fi

if command -v systemctl >/dev/null 2>&1 \
    && systemctl --user show --property=Version >/dev/null 2>&1; then
  echo "==> systemctl --user stop hga-qwen38-gateway.service hga-qwen38.service" >&2
  systemctl --user stop hga-qwen38-gateway.service hga-qwen38.service || true
fi

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
