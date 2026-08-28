#!/usr/bin/env bash
# Back-compat wrapper. GGUF lookup/download lives in setup.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HGA_SETUP_ONLY=gguf
exec bash "$ROOT/scripts/setup.sh"
