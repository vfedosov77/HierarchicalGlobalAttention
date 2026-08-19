#!/usr/bin/env bash
# Fast Qwen3.8-27B inference on dual Xeon Gold 6148 (80 threads) + Tesla V100 16GB.
#
# llama.cpp already runs this architecture (qwen35) with INT4 GGUF and AVX-512.
# It does *not* ship HGA; this wrapper turns it on. Dense softmax attention
# stays on the CPU (--no-kv-offload). FFN / GDN weights go to the V100 as far
# as 16 GB allows (o_proj / attn_out is a GPU tensor when the layer is offloaded).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# turing1: CUDA toolkit was copied to ~/opt/cuda-12.5 (no public internet).
if [[ -d "${CUDA_HOME:-$HOME/opt/cuda-12.5}/lib64" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.5}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
LLAMA="${HGA_LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
export LD_LIBRARY_PATH="${LLAMA}/build/bin:${LD_LIBRARY_PATH:-}"
# llama-completion is the non-REPL generator; llama-cli defaults to chat.
BIN="${LLAMA}/build/bin/llama-completion"
[[ -x "$BIN" ]] || BIN="${LLAMA}/build/bin/llama-cli"
MODEL="${HGA_MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}"
LEVELS="${HGA_LEVELS:-2}"
CTX="${HGA_CTX:-2048}"
T="${HGA_THREADS:-40}"
TB="${HGA_THREADS_BATCH:-80}"
# Do not pass -ngl 99 on a 16 GB card: llama.cpp --fit will abort instead of shrinking.
# Omit ngl so auto-fit packs FFN/GDN into the V100; --no-kv-offload keeps KV+HGA on CPU.
PROMPT="${*:-The capital of France is Paris. Continue with one sentence:}"

if [[ ! -x "$BIN" ]]; then
  echo "build first: $ROOT/scripts/setup.sh" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing model $MODEL — copy Q4_K_M GGUF here (server has no internet)" >&2
  exit 1
fi

NUMA=(--numa distribute)
if [[ "${HGA_NUMA:-1}" == "0" ]]; then NUMA=(); fi
PREC=(--hga-i8)
if [[ "${HGA_I8:-1}" == "0" ]]; then PREC=(--hga-f16); fi

exec "$BIN" \
  -m "$MODEL" \
  -t "$T" -tb "$TB" \
  "${NUMA[@]}" \
  --no-kv-offload \
  --flash-attn on \
  --hga --hga-levels "$LEVELS" \
  "${PREC[@]}" \
  --hga-chunk 64 --hga-group 16 \
  --hga-keep-first 2 --hga-keep-last 8 \
  --hga-frac-l1 0.08 --hga-frac-l2 0.04 \
  -c "$CTX" \
  -n "${HGA_N:-64}" \
  -p "$PROMPT"
