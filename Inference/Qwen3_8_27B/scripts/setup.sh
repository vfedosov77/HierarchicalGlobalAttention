#!/usr/bin/env bash
# Clone llama.cpp, apply HGA, build CPU (AVX-512) llama-cli / llama-bench / llama-server
# and the standalone HGA test/bench.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
LLAMA="${HGA_LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
BUILD_HGA="${ROOT}/build"
BUILD_LLAMA="${LLAMA}/build"
JOBS="${JOBS:-$(nproc)}"
# Physical cores are better for memory-bound INT4 decode on Xeon 6148 (40c/80t).
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

echo "==> building llama.cpp (CPU, native AVX)"
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
  echo "==> CUDA toolkit detected — enabling ggml-cuda (sm_70 for V100)"
  CMAKE_ARGS+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES=70
  )
fi
cmake "${CMAKE_ARGS[@]}"
# llama-cli is built only when LLAMA_BUILD_SERVER=ON (cli lives under tools/server).
cmake --build "$BUILD_LLAMA" -j"$JOBS" --target llama-cli llama-bench

echo
echo "Built:"
echo "  $BUILD_HGA/hga-test"
echo "  $BUILD_HGA/hga-bench"
echo "  $BUILD_LLAMA/bin/llama-cli"
echo "  $BUILD_LLAMA/bin/llama-bench"
echo "  $BUILD_LLAMA/bin/llama-server"
echo
echo "Run CPU tests:  $BUILD_HGA/hga-test"
echo "Run 1L vs 2L:   $BUILD_HGA/hga-bench --threads ${PHYS:-$JOBS}"
