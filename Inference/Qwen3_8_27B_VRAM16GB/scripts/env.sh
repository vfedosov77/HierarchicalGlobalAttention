# Source this on a 16 GB NVIDIA GPU host before running HGA.
# Detects CUDA, physical CPU cores, and the Unsloth UD-Q4_K_M GGUF.
#
#   source scripts/env.sh

_hga_this="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HGA_ROOT="${HGA_ROOT:-$(cd "$_hga_this/.." && pwd)}"

_hga_physical_cores() {
  if command -v lscpu >/dev/null 2>&1; then
    local n
    n="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
    if [[ "${n:-0}" -ge 1 ]]; then
      echo "$n"
      return
    fi
  fi
  nproc
}

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ ! -d "$CUDA_HOME/lib64" && -d /usr/local/cuda/lib64 ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [[ -d "$CUDA_HOME/bin" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

export HGA_BIN="${HGA_BIN:-$HGA_ROOT/third_party/llama.cpp/build/bin}"
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${HGA_BIN}:${LD_LIBRARY_PATH:-}"
export HGA_MODEL="${HGA_MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf}"

# 16 GB packing: drop the 768-token prefill graph after the prompt, unmap
# empty CUDA VMM pools, then pin leftover VERIFY exchange pairs.
export HGA_VMM_KEEP="${HGA_VMM_KEEP:-0}"
export HGA_UBATCH="${HGA_UBATCH:-768}"
export HGA_PREFILL_UBATCH="${HGA_PREFILL_UBATCH:-768}"
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"

# Do not mmap the ~15 GB GGUF into RAM. llama.cpp streams each GPU tensor
# through a 16 MiB scratch buffer (scripts/apply_hga.py).
export HGA_LOAD_MODE="${HGA_LOAD_MODE:-none}"

export HGA_THREADS="${HGA_THREADS:-$(_hga_physical_cores)}"
export HGA_THREADS_BATCH="${HGA_THREADS_BATCH:-1}"
# 2 query × 5 key tiles is a 40-task prefill layout. Leave off unless you
# have that many physical cores (HGA_PREFILL_K_TILES=1).
export HGA_PREFILL_K_TILES="${HGA_PREFILL_K_TILES:-0}"
# L2 GEMV is quarantined; pinning workers to physical cores on SMT hosts
# has crashed local 6-core / 12-thread runs.
export HGA_L2_OFF="${HGA_L2_OFF:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$HGA_THREADS}"
# Using every logical CPU (SMT): bind to threads. Using only physical cores:
# bind to cores so OpenMP does not pair siblings.
if [[ "${HGA_THREADS}" -eq "$(nproc)" ]]; then
  export OMP_PLACES="${OMP_PLACES:-threads}"
else
  export OMP_PLACES="${OMP_PLACES:-cores}"
fi
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export HGA_NUMA="${HGA_NUMA:-0}"

# United GPU prefill: CPU routes every 64-token chunk, CUDA attends one
# historical INT8 image. 3200 keys covers the measured 8K union (~3104).
export HGA_GPU_PREFILL="${HGA_GPU_PREFILL:-1}"
export HGA_GPU_PREFILL_MIN_KEYS="${HGA_GPU_PREFILL_MIN_KEYS:-1552}"
export HGA_GPU_PREFILL_MAX_KEYS="${HGA_GPU_PREFILL_MAX_KEYS:-3200}"
# -ot already places exchange layers; llama.cpp --fit-target cannot also run.
export HGA_FIT_TARGET="${HGA_FIT_TARGET:-0}"
export HGA_VERIFY_STREAMS="${HGA_VERIFY_STREAMS:-2}"
