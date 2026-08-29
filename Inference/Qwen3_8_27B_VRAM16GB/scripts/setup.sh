#!/usr/bin/env bash
# Clone llama.cpp, apply HGA, download the Qwen3.8-27B GGUF, and build CUDA
# llama-cli / llama-bench / llama-server plus the standalone HGA test/bench.
# CUDA toolkit/host gcc are probed (scripts/select_cuda.sh).
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

# Python used for helper packages (huggingface_hub, cmake wheels).
# Do not assume `pip install --user`: that fails when python3 is a venv
# (user site-packages are hidden). Prefer the active env, then --user,
# then a project-local venv under third_party/.
HGA_PYTHON="${HGA_PYTHON:-python3}"
HGA_BOOTSTRAP_VENV="$ROOT/third_party/pyenv"

hga_in_venv() {
  local py="${1:-$HGA_PYTHON}"
  "$py" -c 'import sys; raise SystemExit(0 if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else 1)' 2>/dev/null
}

hga_ensure_pip() {
  local py="${1:-$HGA_PYTHON}"
  if "$py" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  echo "==> bootstrapping pip for $py"
  "$py" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$py" -m pip --version >/dev/null 2>&1
}

# Install Python packages into an env this interpreter can write.
hga_pip_install() {
  local py="${HGA_PYTHON}"
  if ! command -v "$py" >/dev/null 2>&1 && [[ ! -x "$py" ]]; then
    echo "error: python3 not found (needed to install $*)." >&2
    return 1
  fi
  if ! hga_ensure_pip "$py"; then
    echo "error: pip is not available for $py (needed to install $*)." >&2
    return 1
  fi
  if hga_in_venv "$py"; then
    echo "==> pip install (venv $py): $*"
    "$py" -m pip install "$@"
    return
  fi
  echo "==> pip install --user: $*"
  if "$py" -m pip install --user "$@"; then
    export PATH="${HOME}/.local/bin:${PATH}"
    return 0
  fi
  echo "==> pip install (site): $*"
  if "$py" -m pip install "$@"; then
    return 0
  fi
  if [[ ! -x "$HGA_BOOTSTRAP_VENV/bin/python" ]]; then
    echo "==> creating $HGA_BOOTSTRAP_VENV for helper packages"
    "$py" -m venv "$HGA_BOOTSTRAP_VENV"
  fi
  HGA_PYTHON="$HGA_BOOTSTRAP_VENV/bin/python"
  export PATH="$HGA_BOOTSTRAP_VENV/bin:$PATH"
  hga_ensure_pip "$HGA_PYTHON" || return 1
  echo "==> pip install (bootstrap venv): $*"
  "$HGA_PYTHON" -m pip install "$@"
}

# --- GGUF: short local lookup, download only if nothing is on this host ---
# Do not walk the filesystem. Check the places people actually put a 15 GiB
# file (cwd, this tree, ~/models, Downloads, HF hub cache). First complete
# hit wins. UD-Q4_K_M is preferred; any Qwen3.8-27B*.gguf in those dirs is ok.
HGA_GGUF_FILE="${HGA_GGUF_FILE:-Qwen3.8-27B-UD-Q4_K_M.gguf}"
HGA_HF_REPO="${HGA_HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
_HGA_GGUF_MIN_BYTES=1000000000
_HGA_GGUF_STAMP="$ROOT/third_party/gguf.path"

_hga_gguf_ok() {
  local f="$1" sz
  [[ -n "$f" && -f "$f" && -r "$f" ]] || return 1
  sz="$(stat -c%s "$f" 2>/dev/null || echo 0)"
  [[ "$sz" -gt "$_HGA_GGUF_MIN_BYTES" ]]
}

_hga_gguf_realpath() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    readlink -f "$1" 2>/dev/null || echo "$1"
  fi
}

_hga_gguf_consider() {
  local f="$1"
  if _hga_gguf_ok "$f"; then
    HGA_MODEL="$(_hga_gguf_realpath "$f")"
    return 0
  fi
  return 1
}

# One directory, no recursion. Prefer the Unsloth UD-Q4_K_M name.
_hga_gguf_scan_dir() {
  local d="$1" f
  [[ -n "$d" && -d "$d" ]] || return 1
  _hga_gguf_consider "$d/$HGA_GGUF_FILE" && return 0
  _hga_gguf_consider "$d/Qwen3.8-27B-Q4_K_M.gguf" && return 0
  local _old_nullglob
  _old_nullglob="$(shopt -p nullglob || true)"
  shopt -s nullglob
  for f in "$d"/Qwen3.8-27B*.gguf "$d"/Qwen3.8-27B-GGUF/Qwen3.8-27B*.gguf; do
    if _hga_gguf_consider "$f"; then
      eval "$_old_nullglob"
      return 0
    fi
  done
  eval "$_old_nullglob"
  return 1
}

_hga_gguf_find() {
  local d f
  if [[ -n "${HGA_MODEL:-}" ]] && _hga_gguf_consider "$HGA_MODEL"; then
    return 0
  fi
  if [[ -n "${_HGA_GGUF_STAMP:-}" && -f "$_HGA_GGUF_STAMP" ]]; then
    if _hga_gguf_consider "$(cat "$_HGA_GGUF_STAMP")"; then
      return 0
    fi
  fi
  for d in \
    "${PWD:-.}" \
    "$ROOT" \
    "$ROOT/models" \
    "$REPO_ROOT" \
    "$REPO_ROOT/models" \
    "${HGA_MODEL_DIR:-}" \
    "$HOME/models/Qwen3.8-27B-GGUF" \
    "$HOME/models" \
    "$HOME/Downloads" \
    "$HOME" \
    /models \
    /data/models \
    /opt/models
  do
    _hga_gguf_scan_dir "$d" && return 0
  done
  local _old_nullglob
  _old_nullglob="$(shopt -p nullglob || true)"
  shopt -s nullglob
  for f in \
    "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/"*"/$HGA_GGUF_FILE" \
    "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/"*"/Qwen3.8-27B-Q4_K_M.gguf"
  do
    if _hga_gguf_consider "$f"; then
      eval "$_old_nullglob"
      return 0
    fi
  done
  eval "$_old_nullglob"
  return 1
}

_hga_python_has_hf() {
  "$HGA_PYTHON" -c "from huggingface_hub import hf_hub_download" >/dev/null 2>&1
}

_hga_ensure_hf_hub() {
  if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
    return 0
  fi
  if _hga_python_has_hf; then
    return 0
  fi
  echo "==> huggingface_hub not found; installing"
  hga_pip_install "huggingface_hub[cli]"
}

_hga_gguf_http() {
  local dest="$1" file="$2"
  local url="https://huggingface.co/${HGA_HF_REPO}/resolve/main/${file}"
  local part="$dest/${file}.part"
  echo "==> downloading via HTTP $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 2 -C - -o "$part" "$url" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$part" "$url" || return 1
  else
    return 1
  fi
  mv -f "$part" "$dest/$file"
}

_hga_gguf_download() {
  local dest file
  dest="${HGA_MODEL_DIR:-$HOME/models/Qwen3.8-27B-GGUF}"
  file="$HGA_GGUF_FILE"
  mkdir -p "$dest"
  if [[ -f "$dest/$file" ]] && ! _hga_gguf_ok "$dest/$file"; then
    echo "==> removing incomplete $dest/$file ($(stat -c%s "$dest/$file") bytes)"
    rm -f "$dest/$file"
  fi
  echo "==> downloading $HGA_HF_REPO/$file -> $dest (~15.3 GiB)"
  _hga_ensure_hf_hub || true
  if command -v hf >/dev/null 2>&1; then
    hf download "$HGA_HF_REPO" --include "$file" --local-dir "$dest"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$HGA_HF_REPO" "$file" --local-dir "$dest"
  elif _hga_python_has_hf; then
    "$HGA_PYTHON" -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$HGA_HF_REPO', '$file', local_dir='$dest'))"
  elif _hga_gguf_http "$dest" "$file"; then
    :
  else
    echo "error: could not install huggingface_hub and no curl/wget fallback." >&2
    echo "install huggingface_hub or copy a Qwen3.8-27B GGUF into $ROOT or ~/models/." >&2
    return 1
  fi
  if ! _hga_gguf_ok "$dest/$file"; then
    echo "error: $dest/$file missing or too small after download." >&2
    echo "copy a Qwen3.8-27B GGUF into the current directory, $ROOT, or ~/models/" >&2
    echo "and re-run, or set HGA_MODEL to the file." >&2
    return 1
  fi
  HGA_MODEL="$(_hga_gguf_realpath "$dest/$file")"
}

hga_ensure_gguf() {
  echo "==> GGUF (look in cwd, this tree, ~/models, Downloads, HF cache)"
  if _hga_gguf_find; then
    echo "  using existing $HGA_MODEL"
    ls -lh "$HGA_MODEL"
  else
    _hga_gguf_download
  fi
  export HGA_MODEL
  printf '%s\n' "$HGA_MODEL" >"$_HGA_GGUF_STAMP"
}

# deploy.py uses this to resolve weights without rebuilding llama-server.
if [[ "${HGA_SETUP_ONLY:-}" == "gguf" ]]; then
  hga_ensure_gguf
  echo "HGA_MODEL=$HGA_MODEL"
  echo "HGA_SETUP_OK"
  exit 0
fi

# Release pin: HGA is written against llama.cpp v0.3.0.  ggml-org/src on master
# drifts between releases, so cloning master can silently break the patch (new
# signatures in register_kv_cache_unified / graph caches etc.).  Always set up at
# the exact release we validate against.
#
# v0.3.0 is an annotated tag on the same commit as nightly b10621.  A shallow
# clone of v0.3.0 therefore makes `git describe --tags --exact-match` print
# b10621 (the annotated tag object is not a commit).  Pin by commit SHA.
#
# - Missing tree: clone v0.3.0 (or reuse a local pin checkout if already here).
# - Existing git-backed tree at the wrong SHA: ask, then fetch/checkout the pin
#   (clone a fresh tree if fetch cannot).  HGA_LLAMA_SWITCH=yes|no skips the
#   prompt.  No TTY (CI / redirected stdin) defaults to yes so deploy.py works.
# - Copied tree with no .git (offline host): no tag to check, accept as-is.
LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
LLAMA_TAG=v0.3.0
LLAMA_PIN_SHA=c1d0e7a004015f23bc0233470b747b596f29b264
LLAMA_MANAGED="$ROOT/third_party/llama.cpp"

hga_llama_head() {
  git -C "$1" rev-parse HEAD 2>/dev/null || true
}

hga_llama_is_pin() {
  local head tag_sha
  head="$(hga_llama_head "$1")"
  [[ -n "$head" ]] || return 1
  [[ "$head" == "$LLAMA_PIN_SHA" ]] && return 0
  tag_sha="$(git -C "$1" rev-parse -q --verify "refs/tags/${LLAMA_TAG}^{commit}" 2>/dev/null || true)"
  [[ -n "$tag_sha" && "$head" == "$tag_sha" ]]
}

hga_llama_describe() {
  local dir="$1" tag head
  tag="$(git -C "$dir" describe --tags --exact-match 2>/dev/null || true)"
  head="$(hga_llama_head "$dir")"
  if [[ -n "$tag" && -n "$head" ]]; then
    echo "${tag} (${head:0:12})"
  elif [[ -n "$head" ]]; then
    echo "${head:0:12}"
  else
    echo "unknown"
  fi
}

hga_llama_print_pin() {
  local head extra
  head="$(hga_llama_head "$LLAMA")"
  extra="$(git -C "$LLAMA" describe --tags --exact-match 2>/dev/null || true)"
  if [[ -n "$extra" && "$extra" != "$LLAMA_TAG" ]]; then
    echo "  pinned: $LLAMA_TAG (${head:0:12}, git describe=$extra)"
  else
    echo "  pinned: $LLAMA_TAG (${head:0:12})"
  fi
}

hga_llama_online() {
  command -v git >/dev/null 2>&1 || return 1
  GIT_TERMINAL_PROMPT=0 git ls-remote --tags "$LLAMA_REPO" "$LLAMA_TAG" >/dev/null 2>&1
}

hga_llama_is_managed() {
  [[ "$LLAMA" == "$LLAMA_MANAGED" ]]
}

# A previous pin checkout left beside the default path (e.g. llama.cpp-v0.3.0).
hga_llama_local_pin() {
  local d
  for d in \
    "${HGA_LLAMA_PIN_DIR:-}" \
    "$ROOT/third_party/llama.cpp-${LLAMA_TAG}" \
    "$ROOT/third_party/llama.cpp-v0.3.0"
  do
    [[ -n "$d" && "$d" != "$LLAMA" && -f "$d/src/models/qwen35.cpp" ]] || continue
    if [[ -d "$d/.git" ]] && hga_llama_is_pin "$d"; then
      printf '%s\n' "$d"
      return 0
    fi
  done
  return 1
}

hga_llama_clone_into() {
  local dest="$1"
  echo "==> cloning llama.cpp $LLAMA_TAG -> $dest"
  GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$LLAMA_TAG" "$LLAMA_REPO" "$dest" || return 1
  git -C "$dest" tag -f "$LLAMA_TAG" >/dev/null 2>&1 || true
  hga_llama_is_pin "$dest"
}

# Discard local HGA patches; apply_hga.py re-applies them on the pin.
hga_llama_fetch_reset() {
  local dir="$1" pin
  [[ -d "$dir/.git" ]] || return 1
  echo "==> fetching llama.cpp $LLAMA_TAG into $dir"
  GIT_TERMINAL_PROMPT=0 git -C "$dir" fetch --depth 1 --no-recurse-submodules \
    "$LLAMA_REPO" "+refs/tags/${LLAMA_TAG}:refs/tags/${LLAMA_TAG}" || return 1
  pin="$(git -C "$dir" rev-parse -q --verify "refs/tags/${LLAMA_TAG}^{commit}" 2>/dev/null || true)"
  if [[ -z "$pin" ]]; then
    pin="$(git -C "$dir" rev-parse -q --verify FETCH_HEAD 2>/dev/null || true)"
  fi
  [[ -n "$pin" ]] || return 1
  git -C "$dir" reset --hard "$pin" || return 1
  git -C "$dir" clean -fd >/dev/null || true
  git -C "$dir" tag -f "$LLAMA_TAG" >/dev/null 2>&1 || true
  hga_llama_is_pin "$dir"
}

# Install the pin at $LLAMA via temp dir + swap (managed tree only).
hga_llama_install_pin() {
  local src="${1:-}" tmp src_abs
  tmp="${LLAMA}.hga-pin-$$"
  rm -rf "$tmp"
  if [[ -n "$src" ]]; then
    src_abs="$(cd "$src" && pwd)"
    echo "==> cloning llama.cpp $LLAMA_TAG from local $src_abs"
    if ! git clone --depth 1 "file://${src_abs}" "$tmp" || ! hga_llama_is_pin "$tmp"; then
      rm -rf "$tmp"
      echo "==> copying local $LLAMA_TAG tree"
      cp -a "$src_abs" "$tmp" || { rm -rf "$tmp"; return 1; }
    fi
  else
    hga_llama_clone_into "$tmp" || { rm -rf "$tmp"; return 1; }
  fi
  rm -rf "$LLAMA"
  mv "$tmp" "$LLAMA"
  hga_llama_is_pin "$LLAMA"
}

hga_llama_confirm_switch() {
  local observed="$1" ans=""
  case "${HGA_LLAMA_SWITCH:-}" in
    1|y|Y|yes|YES|true|TRUE) return 0 ;;
    0|n|N|no|NO|false|FALSE) return 1 ;;
  esac
  # Read the controlling terminal so a prompt works when deploy.py is the parent.
  if [[ -r /dev/tty && -w /dev/tty ]]; then
    echo
    echo "llama.cpp at $LLAMA is ${observed}"
    echo "HGA requires ${LLAMA_TAG} (${LLAMA_PIN_SHA:0:12})."
    echo "This fetches ${LLAMA_TAG} if needed and replaces the current checkout"
    echo "(local HGA patches are discarded, then re-applied)."
    if ! read -r -p "Switch to ${LLAMA_TAG}? [Y/n] " ans </dev/tty; then
      ans=Y
    fi
    case "${ans:-Y}" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      *) return 1 ;;
    esac
  fi
  echo "==> no TTY; switching llama.cpp to ${LLAMA_TAG} (set HGA_LLAMA_SWITCH=no to refuse)"
  return 0
}

hga_ensure_llama() {
  local local_pin="" observed=""
  mkdir -p "$ROOT/third_party"

  if [[ ! -f "$LLAMA/src/models/qwen35.cpp" ]]; then
    if [[ -e "$LLAMA" ]]; then
      echo "==> removing incomplete llama.cpp tree at $LLAMA"
      rm -rf "$LLAMA"
    fi
    if local_pin="$(hga_llama_local_pin)"; then
      echo "==> using local $LLAMA_TAG checkout at $local_pin"
      hga_llama_install_pin "$local_pin" || true
      if hga_llama_is_pin "$LLAMA"; then
        hga_llama_print_pin
        return 0
      fi
    fi
    if hga_llama_online && hga_llama_clone_into "$LLAMA"; then
      hga_llama_print_pin
      return 0
    fi
    echo "error: llama.cpp not at $LLAMA and no network to clone it." >&2
    echo "copy a $LLAMA_TAG llama.cpp tree to that path (this host has no internet)." >&2
    exit 1
  fi

  echo "==> llama.cpp already present at $LLAMA"

  if [[ ! -d "$LLAMA/.git" ]]; then
    echo "  NOTE: $LLAMA has no .git (offline copy); skipping pin check"
    return 0
  fi

  if hga_llama_is_pin "$LLAMA"; then
    hga_llama_print_pin
    return 0
  fi

  observed="$(hga_llama_describe "$LLAMA")"
  if ! hga_llama_confirm_switch "$observed"; then
    echo "error: $LLAMA is at $observed but $LLAMA_TAG ($LLAMA_PIN_SHA) is required (release pin)." >&2
    echo "       re-run and accept the switch, or set HGA_LLAMA_SWITCH=yes," >&2
    echo "       or point HGA_LLAMA_DIR at a $LLAMA_TAG checkout." >&2
    exit 1
  fi

  if hga_llama_fetch_reset "$LLAMA"; then
    hga_llama_print_pin
    return 0
  fi

  if hga_llama_is_managed; then
    if local_pin="$(hga_llama_local_pin)" && hga_llama_install_pin "$local_pin"; then
      echo "  installed $LLAMA_TAG from $local_pin"
      hga_llama_print_pin
      return 0
    fi
    if hga_llama_online && hga_llama_install_pin; then
      hga_llama_print_pin
      return 0
    fi
  fi

  echo "error: failed to switch $LLAMA to $LLAMA_TAG ($LLAMA_PIN_SHA)." >&2
  if ! hga_llama_is_managed; then
    echo "       HGA_LLAMA_DIR is a custom path; checkout $LLAMA_TAG there or unset it." >&2
  else
    echo "       check network access to $LLAMA_REPO, or copy a $LLAMA_TAG tree to $LLAMA." >&2
  fi
  exit 1
}

hga_ensure_llama

python3 "$ROOT/scripts/apply_hga.py" "$LLAMA"

hga_ensure_gguf

# cmake: portable copy, then PATH, then pip (needs network)
if [[ -x "$ROOT/third_party/cmake/bin/cmake" ]]; then
  export PATH="$ROOT/third_party/cmake/bin:$PATH"
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "==> cmake not found; installing"
  hga_pip_install cmake ninja || true
  export PATH="${HOME}/.local/bin:${PATH}"
  if [[ -x "$HGA_BOOTSTRAP_VENV/bin/cmake" ]]; then
    export PATH="$HGA_BOOTSTRAP_VENV/bin:$PATH"
  fi
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
_hga_select_cuda="$ROOT/scripts/select_cuda.sh"
if [[ ! -f "$_hga_select_cuda" ]]; then
  echo "error: missing $_hga_select_cuda (CUDA toolkit probe)." >&2
  echo "this file ships with the tree; pull/update the Qwen3_8_27B_VRAM16GB checkout." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$_hga_select_cuda"
unset _hga_select_cuda
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
