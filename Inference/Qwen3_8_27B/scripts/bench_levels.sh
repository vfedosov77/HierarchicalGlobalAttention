#!/usr/bin/env bash
# Compare 1-level (~8 % tokens, whole chunks of 64) vs 2-level (~4 % tokens,
# same routed chunks, groups of 16) on this machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
THREADS="${HGA_THREADS:-$(nproc)}"
MAX_SEQ="${HGA_BENCH_SEQ:-32768}"

if [[ ! -x "$BUILD/hga-bench" ]]; then
  cmake -S "$ROOT/cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD" -j"$(nproc)"
fi

echo "== unit tests"
"$BUILD/hga-test"

echo
echo "== 1-level vs 2-level microbench (Qwen3.8-27B attention dims, synthetic QKV, INT8 KV)"
"$BUILD/hga-bench" --threads "$THREADS" --max-seq "$MAX_SEQ" --compare

if [[ -n "${HGA_MODEL:-}" && -x "${HGA_LLAMA_DIR:-$ROOT/third_party/llama.cpp}/build/bin/llama-bench" ]]; then
  BENCH="${HGA_LLAMA_DIR:-$ROOT/third_party/llama.cpp}/build/bin/llama-bench"
  echo
  echo "== llama-bench dense (no HGA) vs HGA 1-level vs HGA 2-level"
  COMMON=(-m "$HGA_MODEL" -t "$THREADS" -ngl "${HGA_NGL:-99}" --no-kv-offload -fa 1 -p 4096,8192,16384 -n 64)
  echo "-- dense --"
  "$BENCH" "${COMMON[@]}"
  echo "-- HGA 1-level --"
  "$BENCH" "${COMMON[@]}" --hga --hga-levels 1
  echo "-- HGA 2-level --"
  "$BENCH" "${COMMON[@]}" --hga --hga-levels 2
fi
