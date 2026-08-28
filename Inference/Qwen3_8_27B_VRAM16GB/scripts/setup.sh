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

# Release pin: HGA is written against llama.cpp v0.3.0.  ggml-org/src on master
# drifts between releases, so cloning master can silently break the patch (new
# signatures in register_kv_cache_unified / graph caches etc.).  Always set up at
# the exact release we validate against.
#
# v0.3.0 is an annotated tag on the same commit as nightly b10621.  A shallow
# clone of v0.3.0 therefore makes `git describe --tags --exact-match` print
# b10621 (the annotated tag object is not a commit).  Pin by commit SHA.
#
# - Fresh clone: fetch v0.3.0 and detach at it.
# - Existing git-backed tree: HEAD must be the pin SHA (or the peeled v0.3.0
#   tag).  A drifted master with some other exact tag is rejected.
# - Copied tree with no .git (offline host): no tag to check, accept as-is.
LLAMA_TAG=v0.3.0
LLAMA_PIN_SHA=c1d0e7a004015f23bc0233470b747b596f29b264

# Offline-friendly: a copied tree is enough. Only clone if git is available and nothing is there.
if [[ ! -f "$LLAMA/src/models/qwen35.cpp" ]]; then
  if command -v git >/dev/null 2>&1 && git ls-remote --tags https://github.com/ggml-org/llama.cpp.git "$LLAMA_TAG" >/dev/null 2>&1; then
    if [[ -e "$LLAMA" ]]; then
      echo "==> removing incomplete llama.cpp tree at $LLAMA"
      rm -rf "$LLAMA"
    fi
    echo "==> cloning llama.cpp $LLAMA_TAG"
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git "$LLAMA"
    git -C "$LLAMA" tag -f "$LLAMA_TAG" >/dev/null 2>&1 || true
  else
    echo "error: llama.cpp not at $LLAMA and no network to clone it." >&2
    echo "copy a $LLAMA_TAG llama.cpp tree to that path (this host has no internet)." >&2
    exit 1
  fi
else
  echo "==> llama.cpp already present at $LLAMA"
fi

# Pin check for a git-backed tree (skipped for offline copied trees with no .git).
if [[ -d "$LLAMA/.git" ]]; then
  _head="$(git -C "$LLAMA" rev-parse HEAD 2>/dev/null || true)"
  _tag_sha="$(git -C "$LLAMA" rev-parse -q --verify "refs/tags/${LLAMA_TAG}^{commit}" 2>/dev/null || true)"
  if [[ -n "$_head" && ( "$_head" == "$LLAMA_PIN_SHA" || ( -n "$_tag_sha" && "$_head" == "$_tag_sha" ) ) ]]; then
    _got_tag="$(git -C "$LLAMA" describe --tags --exact-match 2>/dev/null || true)"
    if [[ -n "$_got_tag" && "$_got_tag" != "$LLAMA_TAG" ]]; then
      echo "  pinned: $LLAMA_TAG (${_head:0:12}, git describe=$_got_tag)"
    else
      echo "  pinned: $LLAMA_TAG (${_head:0:12})"
    fi
  else
    _got_tag="$(git -C "$LLAMA" describe --tags --exact-match 2>/dev/null || echo '')"
    if [[ -n "$_got_tag" ]]; then
      echo "error: $LLAMA is at '$_got_tag' (${_head:0:12}) but $LLAMA_TAG ($LLAMA_PIN_SHA) is required (release pin)." >&2
      echo "       delete $LLAMA and re-run deploy.py (it runs setup.sh when llama-server is missing)" >&2
      exit 1
    fi
    echo "  NOTE: $LLAMA is a git repo but has no exact-release tag." >&2
  fi
  unset _head _tag_sha _got_tag
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

# nvcc + host gcc are probed, not hardcoded. Ubuntu often has a distro
# CUDA 12.4 wrapper at /usr/bin/nvcc (no sm_120) and gcc 15 (too new for
# CUDA 12.8). scripts/select_cuda.sh picks a toolkit that supports this
# machine's GPU and a host compiler that toolkit will actually compile with.
# shellcheck disable=SC1091
source "$ROOT/scripts/select_cuda.sh"
_hga_have_gpu=0
if command -v nvidia-smi >/dev/null 2>&1; then
  _hga_have_gpu=1
fi
_hga_cuda_ok=0
if [[ "$_hga_have_gpu" -eq 1 ]] || command -v nvcc >/dev/null 2>&1 || [[ -n "${CUDA_HOME:-}" ]]; then
  if hga_select_cuda; then
    _hga_cuda_ok=1
    export CUDA_HOME="$HGA_CUDA_HOME"
    export PATH="$HGA_CUDA_HOME/bin:$PATH"
    export CUDACXX="$HGA_NVCC"
  elif [[ "$_hga_have_gpu" -eq 1 ]]; then
    echo "A CPU-only llama-server cannot load this 27B GGUF on 16 GB (CPU repack abort)." >&2
    exit 1
  else
    echo "warning: CUDA toolkit unusable; building CPU-only llama.cpp" >&2
  fi
fi
unset _hga_have_gpu

# ggml compiles AVX-512 CPU kernels into the *default* backend with no runtime
# dispatch: a build with -DGGML_AVX512=ON SIGILLs in ggml_cpu_init() on any CPU
# that lacks avx512f (same class of crash as HGA's own AVX-512 kernels).  Gate it
# the same way: only enable when the build host reports avx512f in /proc/cpuinfo.
GGML_AVX512_FLAG=OFF
if [[ -r /proc/cpuinfo ]] && grep -q ' avx512f' /proc/cpuinfo; then
  GGML_AVX512_FLAG=ON
fi

echo "==> building llama.cpp (ggml-CPU AVX-512: ${GGML_AVX512_FLAG})"
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
  -DGGML_AVX512=${GGML_AVX512_FLAG}
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
if [[ "$_hga_cuda_ok" -eq 1 ]]; then
  if [[ -f "$BUILD_LLAMA/CMakeCache.txt" ]]; then
    _prev_nvcc="$(sed -n 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' "$BUILD_LLAMA/CMakeCache.txt" | head -1)"
    _prev_cxx="$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$BUILD_LLAMA/CMakeCache.txt" | head -1)"
    if [[ "$_prev_nvcc" != "$HGA_NVCC" || "$_prev_cxx" != "$HGA_CUDA_HOST_CXX" ]]; then
      echo "==> clearing llama.cpp CMake cache (was nvcc=${_prev_nvcc:-none} cxx=${_prev_cxx:-none})"
      rm -rf "$BUILD_LLAMA/CMakeCache.txt" "$BUILD_LLAMA/CMakeFiles"
    fi
    unset _prev_nvcc _prev_cxx
  fi
  CMAKE_ARGS+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_COMPILER="$HGA_NVCC"
    -DCMAKE_CUDA_ARCHITECTURES="$HGA_CUDA_ARCH"
    -DCMAKE_CUDA_HOST_COMPILER="$HGA_CUDA_HOST_CXX"
    -DCMAKE_C_COMPILER="$HGA_CUDA_HOST_CC"
    -DCMAKE_CXX_COMPILER="$HGA_CUDA_HOST_CXX"
  )
fi
unset _hga_cuda_ok
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
