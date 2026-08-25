#!/usr/bin/env bash
# Fast Qwen3.8-27B inference on a 16 GB GPU host (default: 12 threads).
#
# Default packing (README "Default packing"):
#   CUDA-resident: every layer except the exchange pairs, including lm_head.
#   CPU mmap: token_embd + streamed exchange layers only
#     (default 8 pairs step-4: blk.0/4/…/28 ↔ 32/36/…/60; HGA_STREAM_2=1 is
#      16/32/48/63; uniform is 10/21/32/42/53/63). MTP blk.64 stays CUDA in HGA_SPEC.
#   Prefill/decode: QKV + lm_head never leave CUDA. Only exchange slots H2D.
#   Decode: q/k-norm ops and their 1 KiB weights on CPU (HGA dataflow),
#     contiguous 24 KiB HGA D2H.
#   Stream pairs default 0↔32 … 28↔60 (8 slots). HGA_STREAM_2=1 is 16↔48/32↔63.
#   Log:  hga-swap: PREFILL  exchange ... lm_head CUDA
#   hga-pin: census after load/PREFILL/DECODE (PUSHED = CUDA0 weight now on
#     CUDA_Host/CPU). HGA_PIN_CHECK=0 off; HGA_PIN_ABORT=1 crash; HGA_PIN_VERBOSE=1
#   Prefill ubatch 768 (one CUDA graph size). After the prompt, drop that
#   graph (n_ubatch=K+1) and unmap the empty VMM pool (do not set
#   HGA_VMM_KEEP=1 — it leaves ~300 MiB mapped and leftover VERIFY pin OOMs).
#   VERIFY streams 2 pairs (16↔48/28↔60); the other 6 are CUDA-resident.
#   HGA_NO_RESIDENT_FFN=1 skips that leftover pin. HGA_SPEC=K → draft-mtp.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/scripts/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/env.sh"
fi
# Independent H2D copy streams are not legal inside a CUDA graph capture.
if [[ -z "${GGML_CUDA_DISABLE_GRAPHS+x}" ]]; then
  export GGML_CUDA_DISABLE_GRAPHS=1
fi
# Prefill keeps the 768-token VMM pool (no unmap between chunks).
# Drop-prefill arms HGA_VMM_UNMAP=1 once so leftover VERIFY pairs fit.
export HGA_VMM_KEEP="${HGA_VMM_KEEP:-0}"
export HGA_UBATCH="${HGA_UBATCH:-768}"
export HGA_PREFILL_UBATCH="${HGA_PREFILL_UBATCH:-768}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${HGA_THREADS:-$(nproc)}}"
export OMP_PLACES="${OMP_PLACES:-threads}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
if [[ "${HGA_VMM_KEEP}" != "0" ]]; then
  echo "warning: HGA_VMM_KEEP=$HGA_VMM_KEEP keeps prefill VMM mapped; leftover VERIFY pin may OOM" >&2
fi
if [[ -n "${HGA_LMHEAD_CPU:-}" && "${HGA_LMHEAD_CPU}" != "0" ]]; then
  echo "error: HGA_LMHEAD_CPU=$HGA_LMHEAD_CPU is not allowed — lm_head must stay on the GPU" >&2
  exit 1
fi
unset HGA_LMHEAD_CPU || true
if [[ -n "${HGA_NO_RESIDENT_FFN:-}" && "${HGA_NO_RESIDENT_FFN}" != "0" ]]; then
  echo "warning: HGA_NO_RESIDENT_FFN=$HGA_NO_RESIDENT_FFN skips leftover VERIFY CUDA pin" >&2
fi
echo "hga: ubatch=${HGA_UBATCH} prefill_ubatch=${HGA_PREFILL_UBATCH} VMM_KEEP=${HGA_VMM_KEEP} leftover_pin=$([ "${HGA_NO_RESIDENT_FFN:-0}" = "1" ] && echo off || echo on) graphs_disabled=${GGML_CUDA_DISABLE_GRAPHS}" >&2
echo "hga-pin-plan: all layers CUDA except exchange slots; lm_head CUDA" >&2
STREAM_BLOCK="${HGA_STREAM_BLOCK:-0}"
VERIFY_STREAMS="${HGA_VERIFY_STREAMS:-2}"
if [[ "$STREAM_BLOCK" != 0 && "$STREAM_BLOCK" != 2 && "$STREAM_BLOCK" != 3 ]]; then
  echo "error: HGA_STREAM_BLOCK=$STREAM_BLOCK must be 0, 2, or 3" >&2
  exit 1
fi
if [[ "$VERIFY_STREAMS" != 2 && "$VERIFY_STREAMS" != 3 ]]; then
  echo "error: HGA_VERIFY_STREAMS=$VERIFY_STREAMS must be 2 or 3" >&2
  exit 1
fi
if [[ "$STREAM_BLOCK" == 2 ]]; then
  echo "hga-pin-plan: PREFILL stream 7 groups  packed 1:2-32:33 + 4-36 8-40 12-44 16-48 20-52 24-56" >&2
  echo "hga-pin-plan: DECODE stream 1 packed group (2 layer pairs) 1:2-32:33; leftover pin CUDA 6 groups" >&2
elif [[ "$STREAM_BLOCK" == 3 ]]; then
  echo "hga-pin-plan: PREFILL stream 6 groups  packed 1:3-32:34 + 4-36 8-40 12-44 16-48 20-52" >&2
  echo "hga-pin-plan: DECODE stream 1 packed group (3 layer pairs) 1:3-32:34; leftover pin CUDA 5 groups" >&2
else
  echo "hga-pin-plan: PREFILL stream 8 pairs  0-32 4-36 8-40 12-44 16-48 20-52 24-56 28-60" >&2
  if [[ "$VERIFY_STREAMS" == 3 ]]; then
    echo "hga-pin-plan: DECODE  stream 3 pairs  0-32 16-48 28-60 ; leftover pin CUDA  4-36 8-40 12-44 20-52 24-56" >&2
  else
    echo "hga-pin-plan: DECODE  stream 2 pairs  16-48 28-60 ; leftover pin CUDA  0-32 4-36 8-40 12-44 20-52 24-56" >&2
  fi
fi
if [[ -d "${CUDA_HOME:-}/lib64" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -d /usr/local/cuda/lib64 ]]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -d "${HOME}/opt/cuda-12.5/lib64" ]]; then
  export CUDA_HOME="${HOME}/opt/cuda-12.5"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
LLAMA="${HGA_LLAMA_DIR:-$ROOT/third_party/llama.cpp}"
export LD_LIBRARY_PATH="${LLAMA}/build/bin:${LD_LIBRARY_PATH:-}"
# llama-completion is the non-REPL generator; llama-cli defaults to chat.
BIN="${LLAMA}/build/bin/llama-completion"
[[ -x "$BIN" ]] || BIN="${LLAMA}/build/bin/llama-cli"
SERVER="${HGA_SERVER:-0}"
if [[ "$SERVER" != "0" ]]; then
  BIN="${LLAMA}/build/bin/llama-server"
  if [[ ! -x "$BIN" ]]; then
    echo "HGA_SERVER requires $BIN — rebuild with scripts/setup.sh" >&2
    exit 1
  fi
fi
MODEL="${HGA_MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf}"
LEVELS="${HGA_LEVELS:-2}"
CTX="${HGA_CTX:-2048}"
# Prefill uses -tb (batch). Decode uses -t. HGA uses physical cores by default.
T="${HGA_THREADS:-$(nproc)}"
# HGA custom ops launch their own OpenMP regions.  Width-3 verify uses
# llama.cpp's batch pool, whose workers otherwise spin while ith=0 runs
# HGA and starve the inner team.  One outer worker leaves the CPUs to HGA.
TB="${HGA_THREADS_BATCH:-1}"
# Do not pass -ngl 99 on a 16 GB card: llama.cpp --fit will abort instead of shrinking.
# Omit ngl so auto-fit packs FFN/GDN into the V100; --no-kv-offload keeps KV+HGA on CPU.
PROMPT_FILE="${HGA_PROMPT_FILE:-}"
PROMPT="${*:-The capital of France is Paris. Continue with one sentence:}"

if [[ ! -x "$BIN" ]]; then
  echo "build first: $ROOT/scripts/setup.sh" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing model $MODEL — copy Qwen3.8-27B-UD-Q4_K_M.gguf here (see scripts/download_qwen38.sh)" >&2
  exit 1
fi
if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
  echo "missing prompt file $PROMPT_FILE" >&2
  exit 1
fi

NUMA=(--numa distribute)
if [[ "${HGA_NUMA:-0}" == "0" ]]; then NUMA=(); fi
PREC=(--hga-i8)
if [[ "${HGA_I8:-1}" == "0" ]]; then PREC=(--hga-f16); fi

# Only exchange layers are host mmap (one CUDA slot per pair, H2D on kick).
# Everything else — dense QKV, lm_head, FFN 31, GDN, o_proj, MTP — loads on
# CUDA0 and must not migrate. The tiny q/k-norm weights remain on CPU because
# decode/verify consumes them there; this avoids a separate scheduler split in
# every full-attention layer. llama.cpp warns that repeated -ot keeps only the
# last value — pass one comma-separated list.
#
# Do NOT add output.weight=CPU. llama.cpp applies -ot with unanchored
# std::regex_search, so a bare 'output.weight' also matches
# blk.N.attn_output.weight and pins all 16 o_proj weights to host memory.
# See README "o_proj on the V100".
STREAM_UNIFORM="${HGA_STREAM_UNIFORM:-${HGA_STREAM_24_56:-0}}"
STREAM_2="${HGA_STREAM_2:-0}"
OT_CPU_COMMON="token_embd\.weight=CPU"
OT_CPU_COMMON="${OT_CPU_COMMON},blk\\..*\\.attn_[qk]_norm\\.weight=CPU"
if [[ -n "${HGA_OT_CPU+x}" ]]; then
  OT_CPU="$HGA_OT_CPU"
elif [[ "$STREAM_BLOCK" == 2 ]]; then
  # One packed VERIFY group plus six single-layer PREFILL-only groups: 16
  # host-mapped layers in total, matching the default placement budget.
  OT_CPU="${OT_CPU_COMMON}"
  for _il in 1 2 32 33 4 36 8 40 12 44 16 48 20 52 24 56; do
    OT_CPU="${OT_CPU},blk\\.${_il}\\..*=CPU"
  done
elif [[ "$STREAM_BLOCK" == 3 ]]; then
  # User-proposed 1..3↔32..34 packed exchange. Five additional pairs keep
  # prefill's total host-mapped layer count at 16.
  OT_CPU="${OT_CPU_COMMON}"
  for _il in 1 2 3 32 33 34 4 36 8 40 12 44 16 48 20 52; do
    OT_CPU="${OT_CPU},blk\\.${_il}\\..*=CPU"
  done
elif [[ "$STREAM_UNIFORM" != "0" ]]; then
  OT_CPU="${OT_CPU_COMMON},blk\.10\..*=CPU,blk\.21\..*=CPU,blk\.32\..*=CPU,blk\.42\..*=CPU,blk\.53\..*=CPU,blk\.63\..*=CPU"
elif [[ "$STREAM_2" != "0" ]]; then
  OT_CPU="${OT_CPU_COMMON},blk\.16\..*=CPU,blk\.32\..*=CPU,blk\.48\..*=CPU,blk\.63\..*=CPU"
else
  # 8 pairs: 0/4/8/12/16/20/24/28 ↔ 32/36/40/44/48/52/56/60
  OT_CPU="${OT_CPU_COMMON}"
  for _il in 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60; do
    OT_CPU="${OT_CPU},blk\\.${_il}\\..*=CPU"
  done
fi

# Speculative MTP: HGA_SPEC=K draft tokens. llama-completion has no spec loop;
# llama-speculative-simple does (--spec-type draft-mtp). Verify batches are
# K+1 tokens and stay in DECODE packing. lm_head stays CUDA. Exchange layers
# are the only weights that H2D.
SPEC=()
SPEC_K=0
if [[ -n "${HGA_SPEC:-}" && "${HGA_SPEC}" != "0" ]]; then
  SPEC_K="${HGA_SPEC}"
  if [[ ! "$SPEC_K" =~ ^[1-9][0-9]*$ ]]; then
    echo "HGA_SPEC must be a positive integer (draft tokens K), got: $SPEC_K" >&2
    exit 1
  fi
  export HGA_SPEC="$SPEC_K"
  if [[ "$SERVER" == "0" ]]; then
    SPEC_BIN="${LLAMA}/build/bin/llama-speculative-simple"
    if [[ ! -x "$SPEC_BIN" ]]; then
      echo "HGA_SPEC requires $SPEC_BIN — rebuild with scripts/setup.sh" >&2
      exit 1
    fi
    BIN="$SPEC_BIN"
  fi
  SPEC=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_K")

  # A confidence gate of 0.8 commonly truncates the one-layer Qwen MTP draft
  # to one token, so HGA_SPEC=2 ends up verifying only two tokens. Speculative
  # verification remains exact when a weak proposal is rejected; use the full
  # configured draft by default and let callers restore a gate if desired.
  SPEC+=(--spec-draft-p-min "${HGA_SPEC_PMIN:-0}")

  if [[ -z "${HGA_TEMP+x}" ]]; then
    SPEC+=(--temp 0)
  fi
fi
# MTP blk.64 stays CUDA with every other non-exchange layer. Do not -ot it to CPU.

if [[ "$STREAM_UNIFORM" != "0" ]]; then
  export HGA_STREAM_UNIFORM=1
fi

# Streamed layers are CPU-mmap'd. Non-exchange weights including lm_head stay
# CUDA. Set HGA_OT_HEADROOM only if OOMs.
HEADROOM="${HGA_OT_HEADROOM-}"

OT=()
if [[ "${HGA_OT:-1}" != "0" ]]; then
  if [[ -n "$HEADROOM" ]]; then
    OT=(-ot "$OT_CPU,$HEADROOM")
  else
    OT=(-ot "$OT_CPU")
  fi
fi

FIT=()
# --fit-target aborts when -ot overrides are set ("already set by user").
# HGA always pins exchange layers with -ot, so fit cannot pack the 16 GB
# card and must stay off unless the caller disables -ot.
if [[ "${HGA_OT:-1}" != "0" ]]; then
  :
elif [[ "${HGA_FIT_TARGET:-0}" != "0" ]]; then
  FIT=(--fit-target "${HGA_FIT_TARGET}")
fi

LOAD=()
if [[ -n "${HGA_LOAD_MODE:-none}" ]]; then
  LOAD=(--load-mode "${HGA_LOAD_MODE:-none}")
fi

OPOFF=()
if [[ "${HGA_NO_OP_OFFLOAD:-1}" != "0" ]]; then
  OPOFF=(--no-op-offload)
fi

VERBOSE=()
if [[ "${HGA_VERBOSE:-0}" != "0" ]]; then
  VERBOSE=(-v)
fi

# llama-server enables recurrent-state context checkpoints by default.  They
# deliberately split the end of a prompt at 4+n_ubatch and 4 tokens so an
# edited/branched prompt can restore an older hybrid state.  HGA's production
# server traffic is append-only and retains the exact slot prefix, so those
# checkpoints add short mid-prompt evals and unnecessary PREFILL/VERIFY graph
# transitions without improving cache reuse.  Keep them opt-in for workloads
# that need prompt branching.
CTX_CHECKPOINTS=()
if [[ "$SERVER" != "0" ]]; then
  HGA_CTX_CHECKPOINTS="${HGA_CTX_CHECKPOINTS:-0}"
  if [[ ! "$HGA_CTX_CHECKPOINTS" =~ ^[0-9]+$ ]]; then
    echo "error: HGA_CTX_CHECKPOINTS=$HGA_CTX_CHECKPOINTS must be a non-negative integer" >&2
    exit 1
  fi
  CTX_CHECKPOINTS=(--ctx-checkpoints "$HGA_CTX_CHECKPOINTS")
  echo "hga-server: context checkpoints=$HGA_CTX_CHECKPOINTS (0=append-only default)" >&2

  # Index prompt families by their first 256 tokens. Recurrent state is saved
  # at completed turns and at useful suffix intersections; ordinary first
  # prefill batches are never split merely to populate the cache.
  export HGA_LAZY_PREFIX_CACHE="${HGA_LAZY_PREFIX_CACHE:-8}"
  if [[ ! "$HGA_LAZY_PREFIX_CACHE" =~ ^[0-9]+$ ]]; then
    echo "error: HGA_LAZY_PREFIX_CACHE=$HGA_LAZY_PREFIX_CACHE must be a non-negative integer" >&2
    exit 1
  fi
  echo "hga-server: lazy prefix cache=$HGA_LAZY_PREFIX_CACHE (0=disabled, 256-token hash key)" >&2
fi

# Extra llama.cpp flags, space-separated (e.g. HGA_EXTRA='-no-cnv --reasoning off')
EXTRA=()
if [[ -n "${HGA_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=(${HGA_EXTRA})
fi

PROMPT_ARG=()
if [[ "$SERVER" != "0" ]]; then
  # Requests supply their prompts through the HTTP API.  In particular, do
  # not give the server a startup prompt: the mixed-turn benchmark must own
  # slot 0 from its first prefill onward.
  :
elif [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_ARG=(-f "$PROMPT_FILE")
else
  PROMPT_ARG=(-p "$PROMPT")
fi

exec "$BIN" \
  -m "$MODEL" \
  -t "$T" -tb "$TB" \
  "${NUMA[@]}" \
  --no-kv-offload --offload-rs \
  --flash-attn on \
  --hga --hga-levels "$LEVELS" \
  ${HGA_WAVE:+--hga-wave} \
  "${PREC[@]}" \
  --hga-chunk 64 --hga-group 16 \
  --hga-keep-first "${HGA_KEEP_FIRST:-2}" --hga-keep-last "${HGA_KEEP_LAST:-7}" \
  --hga-frac-l1 0.08 --hga-frac-l2 0.04 \
  "${FIT[@]}" \
  "${LOAD[@]}" \
  "${OPOFF[@]}" \
  "${OT[@]}" \
  "${VERBOSE[@]}" \
  "${SPEC[@]}" \
  "${CTX_CHECKPOINTS[@]}" \
  "${EXTRA[@]}" \
  -ub "$HGA_UBATCH" \
  -b "${HGA_BATCH:-$CTX}" \
  -c "$CTX" \
  -n "${HGA_N:-64}" \
  ${HGA_TEMP:+--temp "$HGA_TEMP"} \
  "${PROMPT_ARG[@]}"
