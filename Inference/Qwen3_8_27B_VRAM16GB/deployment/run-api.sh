#!/usr/bin/env bash
# Runtime settings for the 16 GB GPU host. Invoked by hga-qwen38.service
# or started locally for agent quality tests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/scripts/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/env.sh"
fi
: "${HGA_API_KEY:?HGA_API_KEY must be set}"

export HGA_SERVER=1
export HGA_MODEL="${HGA_MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf}"
# Recommended AccessPoint context is 128K. KV/HGA stay on CPU; GPU holds weights.
export HGA_CTX="${HGA_CTX:-131072}"
export HGA_BATCH=768
# Fixed-shape GPU prefill uses one twelve-chunk block. Historical capacity is
# reserved once; short suffixes pad direct K/V. The binary rebuilds while
# valid history grows, then reuses after saturation — reusing the
# position-zero graph across changing history breaks retrieval.
# Speculative MTP: K draft tokens (verify batch is K+1). Default K=3 for a
# speed A/B vs K=2. Set HGA_SPEC=0 if leftover VERIFY pin OOMs.
export HGA_UBATCH=768
export HGA_PREFILL_UBATCH=768
export HGA_N="${HGA_N:-256}"
export HGA_SPEC="${HGA_SPEC:-2}"
# Prefer deploy.py's measured pick (api.env / cpu_threads.env via env.sh).
export HGA_THREADS="${HGA_THREADS:-$(nproc)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$HGA_THREADS}"
export HGA_THREADS_BATCH=1
export HGA_VERIFY_BATCH=1
export HGA_VERIFY_TILES=1
export HGA_VERIFY_ROWS=0
export HGA_PREFILL_K_TILES="${HGA_PREFILL_K_TILES:-0}"
export HGA_GPU_PREFILL="${HGA_GPU_PREFILL:-1}"
export HGA_GPU_PREFILL_MIN_KEYS="${HGA_GPU_PREFILL_MIN_KEYS:-1552}"
export HGA_GPU_PREFILL_MAX_KEYS="${HGA_GPU_PREFILL_MAX_KEYS:-3200}"
# Experimental HGA activation transport. The default F32 graph is preserved;
# set to 1 to cast GPU<->CPU HGA boundary tensors to F16 on the sending side.
export HGA_F16_TRANSPORT="${HGA_F16_TRANSPORT:-0}"
# Quantized CUDA GEMMs already choose their native fast kernels. Make the
# cuBLAS fallback policy explicit for controlled A/B runs.
export GGML_CUDA_CUBLAS_COMPUTE_TYPE="${GGML_CUDA_CUBLAS_COMPUTE_TYPE:-auto}"
export HGA_PREFILL_STREAM_ASYNC=0
export HGA_PREFILL_STREAM_PACED=0
export HGA_PREFILL_STREAMS="${HGA_PREFILL_STREAMS:-5}"
export HGA_STREAM_ASYNC=1
export HGA_STREAM_PACED=1
export HGA_VERIFY_STREAMS="${HGA_VERIFY_STREAMS:-2}"
export HGA_STREAM_BLOCK=0
export HGA_PIN_CHECK=1
export HGA_CTX_CHECKPOINTS=0
export HGA_LAZY_PREFIX_CACHE="${HGA_LAZY_PREFIX_CACHE:-8}"
export GGML_CUDA_DISABLE_GRAPHS=1
# ggml abort otherwise spawns gdb for a backtrace and looks like a hang.
export GGML_NO_BACKTRACE="${GGML_NO_BACKTRACE:-1}"

# llama-server implements the OpenAI endpoints itself.  Keep one slot: the HGA
# cache is deliberately tuned for one persistent, long-running agent session.
# The public endpoint is api_gateway.py.  Keep the native server loopback-only
# and let the gateway select its per-request thinking/cache profile.
# Unlike the fixed-length speed benchmark, an API server must honor EOS. Tool
# calls use EOS to yield control to the client; --ignore-eos makes Qwen emit
# another assistant/tool message and eventually run to the token limit.
# Keep the key out of `ps` / systemd status. llama-server reads LLAMA_API_KEY.
export LLAMA_API_KEY="${HGA_API_KEY}"
export HGA_EXTRA="--host 127.0.0.1 --port ${HGA_BACKEND_PORT:-8081} --parallel 1 --no-warmup --no-context-shift --jinja --alias qwen3.8-27b-hga"

exec "$ROOT/scripts/run_hga.sh"
