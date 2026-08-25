#!/usr/bin/env bash
# Clone llama.cpp, apply HGA, build CUDA llama-cli / llama-bench / llama-server
# and the standalone HGA test/bench. CUDA arch is taken from nvidia-smi.
set -euo pipefail
trap 'echo "HGA_SETUP_FAIL: line $LINENO exited $?" >&2' ERR

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
LLAMA="${HGA_LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
BUILD_HGA="${ROOT}/build"
BUILD_LLAMA="${LLAMA}/build"
JOBS="${JOBS:-$(nproc)}"
PHYS="${HGA_PHYS_CORES:-}"

mkdir -p "$ROOT/third_party"

# Offline-friendly: a copied tree is enough. Only clone if git is available and nothing is there.
if [[ ! -f "$LLAMA/src/models/qwen35.cpp" ]]; then
  if [[ -d "$LLAMA/.git" ]]; then
    echo "==> llama.cpp present but incomplete at $LLAMA"
  elif command -v git >/dev/null 2>&1 && git ls-remote https://github.com/ggml-org/llama.cpp.git HEAD >/dev/null 2>&1; then
    echo "==> cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA"
  else
    echo "error: llama.cpp not at $LLAMA and no network to clone it." >&2
    echo "copy a patched llama.cpp tree to that path (this host has no internet)." >&2
    exit 1
  fi
else
  echo "==> llama.cpp already present at $LLAMA"
fi

python3 "$ROOT/scripts/apply_hga.py" "$LLAMA"

# cmake: portable copy, then PATH, then pip (needs network)
if [[ -x "$ROOT/third_party/cmake/bin/cmake" ]]; then
  export PATH="$ROOT/third_party/cmake/bin:$PATH"
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "==> installing cmake via pip --user"
  python3 -m pip install --user -q cmake ninja || true
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake not found. copy Kitware cmake linux-x86_64 into third_party/cmake/" >&2
  exit 1
fi

echo "==> building standalone HGA (tests + microbench)"
cmake -S "$ROOT/cpp" -B "$BUILD_HGA" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_HGA" -j"$JOBS"

# Desktop/IDE shells often have no nvcc on PATH. The toolkit still lives
# at /usr/local/cuda on a typical 16 GB CUDA host.
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  for _cuda in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda-12; do
    if [[ -x "${_cuda}/bin/nvcc" ]]; then
      export CUDA_HOME="${_cuda}"
      export PATH="${_cuda}/bin:${PATH}"
      break
    fi
  done
  unset _cuda
fi
if command -v nvidia-smi >/dev/null 2>&1 && ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvidia-smi is present but nvcc was not found. Install the CUDA toolkit or set CUDA_HOME." >&2
  echo "A CPU-only llama-server cannot load this 27B GGUF on 16 GB (CPU repack abort)." >&2
  exit 1
fi

echo "==> building llama.cpp"
CMAKE_ARGS=(
  -S "$LLAMA"
  -B "$BUILD_LLAMA"
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_NATIVE=OFF
  -DGGML_CCACHE=OFF
  -DGGML_AVX=ON
  -DGGML_AVX2=ON
  -DGGML_FMA=ON
  -DGGML_F16C=ON
  -DGGML_AVX512=ON
  -DGGML_AVX512_VNNI=OFF
  -DGGML_CUDA=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_UI=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_OPENSSL=OFF
)
if command -v nvcc >/dev/null 2>&1 || [[ -n "${CUDA_HOME:-}" ]]; then
  CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-}"
  if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  fi
  CUDA_ARCH="${CUDA_ARCH:-86}"
  echo "==> CUDA toolkit detected — enabling ggml-cuda (sm_${CUDA_ARCH}) CUDA_HOME=${CUDA_HOME:-}"
  CMAKE_ARGS+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  )
fi
cmake "${CMAKE_ARGS[@]}"
# llama-cli is built only when LLAMA_BUILD_SERVER=ON (cli lives under tools/server).
# llama-speculative-simple is the one-shot MTP loop used by run_hga.sh HGA_SPEC=K
# (llama-completion does not run common_speculative). apply_hga.py adds that
# target even with LLAMA_BUILD_EXAMPLES=OFF.
cmake --build "$BUILD_LLAMA" -j"$JOBS" --target llama-cli llama-bench llama-completion llama-speculative-simple llama-server

echo
echo "Built:"
echo "  $BUILD_HGA/hga-test"
echo "  $BUILD_HGA/hga-bench"
echo "  $BUILD_LLAMA/bin/llama-cli"
echo "  $BUILD_LLAMA/bin/llama-bench"
echo "  $BUILD_LLAMA/bin/llama-completion"
echo "  $BUILD_LLAMA/bin/llama-server"
echo "  $BUILD_LLAMA/bin/llama-speculative-simple"
echo
echo "CPU tests:      $BUILD_HGA/hga-test"
echo "local e2e:      source $ROOT/scripts/env.sh && python3 $ROOT/tools/bench.py"
echo "HGA_SETUP_OK"
