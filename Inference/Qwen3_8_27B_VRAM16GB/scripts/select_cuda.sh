#!/usr/bin/env bash
# Pick an nvcc + host compiler that can actually build for this machine's GPUs.
#
# Why probing (not a hardcoded toolkit / gcc):
# - Ubuntu's `nvidia-cuda-toolkit` package puts a CUDA 12.4 wrapper at
#   /usr/bin/nvcc. CMake prefers it. That binary has no Blackwell (sm_120).
# - A newer official toolkit often sits in /usr/local/cuda-* on the same box.
# - nvcc rejects a host gcc newer than it shipped with (CUDA 12.8 => gcc <= 14).
#   Default gcc 15 on Ubuntu 25/26 then fails CMake's CUDA compiler test.
# - The inverse also happens: CUDA 12.4 + gcc 11 on an Ampere card is fine.
#   Forcing gcc-14 or CUDA 12.8 would break those hosts.
#
# Strategy: list every nvcc, keep those that advertise the GPU's sm_XX, then
# probe host compilers (user CXX, default g++, then gcc-16..11, clang++) until
# a real device compile of a trivial .cu succeeds. Prefer CUDA_HOME when it
# works; otherwise the newest working toolkit.
#
# Sourced by setup.sh. Standalone: `scripts/select_cuda.sh` prints the pick.

hga_realpath() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  elif command -v readlink >/dev/null 2>&1; then
    readlink -f "$1" 2>/dev/null || echo "$1"
  else
    echo "$1"
  fi
}

hga_nvcc_version() {
  local v
  v=$("$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)
  echo "${v:-0}"
}

hga_version_key() {
  echo "$1" | awk -F. '{printf "%03d%03d%03d\n", $1+0, $2+0, $3+0}'
}

# CMake wants 86, 89, 120. nvidia-smi prints 8.6, 8.9, 12.0.
hga_detect_cuda_arch() {
  local arch
  arch="${CMAKE_CUDA_ARCHITECTURES:-}"
  if [[ -z "$arch" ]]; then
    arch="${HGA_CUDA_ARCH:-}"
  fi
  if [[ -z "$arch" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | tr -d ' .' | awk 'NF' | sort -u | paste -sd ';' - || true)"
  fi
  if [[ -z "$arch" ]]; then
    echo "error: cannot detect GPU compute capability. set CMAKE_CUDA_ARCHITECTURES (e.g. 86, 89, 120)." >&2
    return 1
  fi
  HGA_CUDA_ARCH="$arch"
}

hga_arch_hint() {
  case "$1" in
    120*|121*|100*|101*|103*) echo "CUDA 12.8+ (Blackwell)" ;;
    90*) echo "CUDA 12.0+" ;;
    89*) echo "CUDA 11.8+" ;;
    87*|86*) echo "CUDA 11.1+" ;;
    80*) echo "CUDA 11.0+" ;;
    *) echo "a CUDA toolkit that supports sm_$1" ;;
  esac
}

hga_nvcc_sms() {
  "$1" --help 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u
}

hga_nvcc_supports_archs() {
  local nvcc="$1" archs="$2" sms a
  sms="$(hga_nvcc_sms "$nvcc")"
  if [[ -z "$sms" ]]; then
    return 1
  fi
  local IFS=';'
  # shellcheck disable=SC2086
  set -- $archs
  for a in "$@"; do
    [[ -n "$a" ]] || continue
    echo "$sms" | grep -qx "sm_${a}" || return 1
  done
  return 0
}

hga_cxx_to_cc() {
  local cxx="$1" base dir cc
  base="$(basename "$cxx")"
  dir="$(dirname "$cxx")"
  cc="$base"
  cc="${cc/clang++/clang}"
  cc="${cc/g++/gcc}"
  if [[ "$cc" == "c++" ]]; then
    cc="cc"
  fi
  if [[ -x "$dir/$cc" ]]; then
    echo "$dir/$cc"
    return 0
  fi
  command -v "$cc" 2>/dev/null || return 1
}

_hga_push_nvcc() {
  local cand="$1" r
  [[ -n "$cand" && -x "$cand" ]] || return 0
  r="$(hga_realpath "$cand")"
  [[ -n "${_hga_nvcc_seen[$r]:-}" ]] && return 0
  _hga_nvcc_seen[$r]=1
  _hga_nvccs+=("$r")
}

hga_collect_nvccs() {
  _hga_nvccs=()
  declare -gA _hga_nvcc_seen=()
  _hga_nvcc_seen=()
  local d
  _hga_push_nvcc "${CUDA_HOME:+$CUDA_HOME/bin/nvcc}"
  _hga_push_nvcc "${CUDACXX:-}"
  _hga_push_nvcc "${CMAKE_CUDA_COMPILER:-}"
  if command -v nvcc >/dev/null 2>&1; then
    _hga_push_nvcc "$(command -v nvcc)"
  fi
  for d in \
    /usr/local/cuda \
    /usr/local/cuda-12 \
    /usr/local/cuda-* \
    /usr/lib/cuda \
    /usr/lib/nvidia-cuda-toolkit \
    /opt/cuda \
    /opt/cuda-* \
    "${HOME}/opt/cuda" \
    "${HOME}"/opt/cuda-*
  do
    _hga_push_nvcc "$d/bin/nvcc"
  done
}

hga_sorted_nvccs() {
  local p v
  for p in "${_hga_nvccs[@]}"; do
    v="$(hga_nvcc_version "$p")"
    printf '%s\t%s\n' "$(hga_version_key "$v")" "$p"
  done | sort -r | cut -f2-
}

_hga_emit_cxx() {
  local cand="$1" resolved r
  [[ -n "$cand" ]] || return 0
  if [[ -x "$cand" ]]; then
    resolved="$cand"
  else
    resolved="$(command -v "$cand" 2>/dev/null || true)"
  fi
  [[ -n "$resolved" && -x "$resolved" ]] || return 0
  r="$(hga_realpath "$resolved")"
  [[ -n "${_hga_cxx_seen[$r]:-}" ]] && return 0
  _hga_cxx_seen[$r]=1
  echo "$r"
}

hga_host_cxx_candidates() {
  declare -gA _hga_cxx_seen=()
  _hga_cxx_seen=()
  local n
  _hga_emit_cxx "${CUDAHOSTCXX:-}"
  _hga_emit_cxx "${CXX:-}"
  _hga_emit_cxx "g++"
  _hga_emit_cxx "c++"
  for n in 16 15 14 13 12 11; do
    _hga_emit_cxx "g++-$n"
  done
  _hga_emit_cxx "clang++"
}

hga_probe_nvcc_host() {
  local nvcc="$1" host="$2" archs="$3"
  local tmp a
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/hga-cuda-probe.XXXXXX")" || return 1
  echo 'int main(){return 0;}' >"$tmp/t.cu"
  local args=()
  local IFS=';'
  # shellcheck disable=SC2086
  set -- $archs
  for a in "$@"; do
    [[ -n "$a" ]] || continue
    args+=(--generate-code="arch=compute_${a},code=[compute_${a},sm_${a}]")
  done
  if "$nvcc" -forward-unknown-to-host-compiler -ccbin "$host" \
      "${args[@]}" -c "$tmp/t.cu" -o "$tmp/t.o" >/dev/null 2>&1; then
    rm -rf "$tmp"
    return 0
  fi
  rm -rf "$tmp"
  return 1
}

hga_try_hosts() {
  local nvcc="$1" archs="$2" host cc
  while IFS= read -r host; do
    [[ -n "$host" ]] || continue
    cc="$(hga_cxx_to_cc "$host" || true)"
    if [[ -z "$cc" || ! -x "$cc" ]]; then
      continue
    fi
    if hga_probe_nvcc_host "$nvcc" "$host" "$archs"; then
      HGA_CUDA_HOST_CXX="$host"
      HGA_CUDA_HOST_CC="$cc"
      return 0
    fi
  done < <(hga_host_cxx_candidates)
  return 1
}

# Sets HGA_NVCC, HGA_CUDA_HOME, HGA_CUDA_VERSION, HGA_CUDA_ARCH,
# HGA_CUDA_HOST_CXX, HGA_CUDA_HOST_CC.
hga_select_cuda() {
  HGA_NVCC=""
  HGA_CUDA_HOME=""
  HGA_CUDA_VERSION=""
  HGA_CUDA_HOST_CXX=""
  HGA_CUDA_HOST_CC=""

  hga_detect_cuda_arch || return 1
  hga_collect_nvccs
  if [[ ${#_hga_nvccs[@]} -eq 0 ]]; then
    echo "error: no nvcc found. Install the CUDA toolkit (matching this GPU) or set CUDA_HOME." >&2
    echo "A CPU-only llama-server cannot load this 27B GGUF on 16 GB (CPU repack abort)." >&2
    return 1
  fi

  local preferred=""
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    preferred="$(hga_realpath "${CUDA_HOME}/bin/nvcc")"
  fi

  local -a order=()
  local p
  if [[ -n "$preferred" ]]; then
    order+=("$preferred")
  fi
  while IFS= read -r p; do
    [[ -n "$p" ]] || continue
    [[ "$p" == "$preferred" ]] && continue
    order+=("$p")
  done < <(hga_sorted_nvccs)

  local notes="" nvcc ver
  for nvcc in "${order[@]}"; do
    ver="$(hga_nvcc_version "$nvcc")"
    if ! hga_nvcc_supports_archs "$nvcc" "$HGA_CUDA_ARCH"; then
      notes="${notes}  skip ${nvcc} (${ver}): no sm_${HGA_CUDA_ARCH//;/, sm_}"$'\n'
      continue
    fi
    if hga_try_hosts "$nvcc" "$HGA_CUDA_ARCH"; then
      HGA_NVCC="$nvcc"
      HGA_CUDA_HOME="$(cd "$(dirname "$nvcc")/.." && pwd)"
      HGA_CUDA_VERSION="$ver"
      echo "==> CUDA: nvcc ${ver} (${nvcc})"
      echo "    arch sm_${HGA_CUDA_ARCH//;/, sm_}  host ${HGA_CUDA_HOST_CXX}  cc ${HGA_CUDA_HOST_CC}"
      if [[ -n "$notes" ]]; then
        printf '%s' "$notes"
      fi
      return 0
    fi
    notes="${notes}  skip ${nvcc} (${ver}): sm_${HGA_CUDA_ARCH} ok, no compatible host C++ compiler"$'\n'
  done

  echo "error: no nvcc + host compiler can build for GPU arch sm_${HGA_CUDA_ARCH}." >&2
  printf '%s' "$notes" >&2
  local a hint
  local IFS=';'
  # shellcheck disable=SC2086
  set -- $HGA_CUDA_ARCH
  for a in "$@"; do
    hint="$(hga_arch_hint "$a")"
    echo "  sm_${a} typically needs ${hint}." >&2
  done
  echo "  Install a matching CUDA toolkit and a host gcc that toolkit accepts" >&2
  echo "  (or set CUDA_HOME and CUDAHOSTCXX). Debian/Ubuntu /usr/bin/nvcc is often" >&2
  echo "  an older distro package; a newer toolkit in /usr/local/cuda-* is OK." >&2
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  hga_select_cuda
  echo "HGA_NVCC=$HGA_NVCC"
  echo "HGA_CUDA_HOME=$HGA_CUDA_HOME"
  echo "HGA_CUDA_VERSION=$HGA_CUDA_VERSION"
  echo "HGA_CUDA_ARCH=$HGA_CUDA_ARCH"
  echo "HGA_CUDA_HOST_CXX=$HGA_CUDA_HOST_CXX"
  echo "HGA_CUDA_HOST_CC=$HGA_CUDA_HOST_CC"
fi
