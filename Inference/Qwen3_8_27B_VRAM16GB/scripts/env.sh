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

# deploy.py writes a measured HGA_THREADS pick here. Honor an explicit
# export first, then the calibration, then physical cores.
_HGA_CALIB="${XDG_CONFIG_HOME:-$HOME/.config}/hga-qwen38/cpu_threads.env"
if [[ -z "${HGA_THREADS:-}" && -f "$_HGA_CALIB" ]]; then
  # shellcheck disable=SC1090
  set -a
  . "$_HGA_CALIB"
  set +a
fi
unset _HGA_CALIB
export HGA_THREADS="${HGA_THREADS:-$(_hga_physical_cores)}"
_hga_pack_default="$(_hga_physical_cores)"
if [[ "${_hga_pack_default}" -gt 12 ]]; then
  _hga_pack_default=12
fi
export HGA_PACK_THREADS="${HGA_PACK_THREADS:-$_hga_pack_default}"
unset _hga_pack_default
export HGA_THREADS_BATCH="${HGA_THREADS_BATCH:-1}"
# 2 query × 5 key tiles is a 40-task prefill layout. Leave off unless you
# have that many physical cores (HGA_PREFILL_K_TILES=1).
export HGA_PREFILL_K_TILES="${HGA_PREFILL_K_TILES:-0}"
# L2 GEMV is quarantined; pinning workers to physical cores on SMT hosts
# has crashed local 6-core / 12-thread runs.
export HGA_L2_OFF="${HGA_L2_OFF:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$HGA_THREADS}"
# SMT siblings: more workers than physical cores. Otherwise pin to cores so
# OpenMP does not put two workers on one core while others sit idle.
_hga_phys="$(_hga_physical_cores)"
if [[ "${HGA_THREADS}" -gt "${_hga_phys}" ]]; then
  export OMP_PLACES="${OMP_PLACES:-threads}"
else
  export OMP_PLACES="${OMP_PLACES:-cores}"
fi
unset _hga_phys
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export HGA_NUMA="${HGA_NUMA:-0}"

# United GPU prefill: CPU routes every 64-token chunk, CUDA attends one
# historical INT8 image. 3200 keys covers the measured 8K union (~3104).
export HGA_GPU_PREFILL="${HGA_GPU_PREFILL:-1}"
export HGA_GPU_PREFILL_MIN_KEYS="${HGA_GPU_PREFILL_MIN_KEYS:-1552}"
export HGA_GPU_PREFILL_MAX_KEYS="${HGA_GPU_PREFILL_MAX_KEYS:-3200}"
# Optional CUDA Q8_0 K/V wire for controlled A/B. Routing Q and Kraw retain
# F16 transport; persistent HGA route IDs/cache remain integer.
export HGA_GPU_KV_I8="${HGA_GPU_KV_I8:-0}"
# VERIFY keeps HGA routing on the CPU but stages selected INT8 history for
# CUDA flash attention. Set HGA_GPU_VERIFY=0 for the former CPU-attention A/B.
export HGA_GPU_VERIFY="${HGA_GPU_VERIFY:-1}"
export HGA_GPU_VERIFY_MAX_KEYS="${HGA_GPU_VERIFY_MAX_KEYS:-$HGA_GPU_PREFILL_MAX_KEYS}"
# -ot already places exchange layers; llama.cpp --fit-target cannot also run.
export HGA_FIT_TARGET="${HGA_FIT_TARGET:-0}"
export HGA_VERIFY_STREAMS="${HGA_VERIFY_STREAMS:-2}"
