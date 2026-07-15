#!/usr/bin/env bash
# Ornith-1.0-9B V3-1 matched-context fine-tune (Target B: 32GB RTX 5090, Blackwell SM120).
#
# routed vs dense at the reference seq 2048 — the 32GB card fits the non-quantized bf16 body
# (~18.5 GB) plus the routed-assembly activations (Ornith head_dim=256), so this restores the
# reference matched-context length that the 16GB QLoRA run has to shrink to 1536.  Plain LoRA on a
# bf16 base (NOT QLoRA): --quantization none skips the 4-bit BitsAndBytesConfig and k-bit prep;
# omitting --fp16 selects bf16 compute (no GradScaler — bf16 needs none).  topk_chunks stays at the
# locked default 16 (== 16/32 chunks == the reference 50% selection budget at 2048).  Single
# full-sequence forward, seed 1337, 32 PG19 validation blocks.
#
# NOTE: the non-quantized bf16 path has NOT been run on hardware here (no 32GB card available); it
# is the toggle counterpart of the validated 16GB QLoRA path.  Validate it first with --smoke:
#     .venv/bin/python -m ExistingModelFineTuning.OrnithLongContext.finetune_ornith_qlora_routed \
#         --smoke --quantization none --model deepreinforce-ai/Ornith-1.0-9B
# If 2048 OOMs on your card, drop --seq-len (e.g. 1536) or fall back to --quantization nf4 --fp16.
#
# Usage (launches a detached tmux session that survives disconnects):
#     ExistingModelFineTuning/OrnithLongContext/Scripts/run_ornith_32gb_2k.sh
#     tmux attach -t ornith_32gb            # watch live
#     bash .../run_ornith_32gb_2k.sh --run     # run in the foreground instead (no tmux)
set -u

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SESSION=ornith_32gb

# Detach into tmux unless already re-invoked with --run (the actual work).
if [ "${1:-}" != "--run" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found; install it or run in the foreground: bash '$SCRIPT' --run" >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists (attach: tmux attach -t $SESSION)." >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "bash '$SCRIPT' --run"
  echo "Launched detached tmux session '$SESSION'."
  echo "  attach: tmux attach -t $SESSION"
  echo "  logs:   tail -f $REPO_ROOT/logs/ornith9b_routed_2k.log"
  exit 0
fi

# ---- actual work (runs inside tmux) --------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HF xet blob CDN can 403 on PG19 parquet streaming; if TrainData/pg19_local holds the validation
# shard it is used (deterministic), otherwise validation streams from the Hub like the train split.
export HGA_LOCAL_DATA_DIR=TrainData/pg19_local
mkdir -p logs

for MODE in routed dense; do
  LOG="logs/ornith9b_${MODE}_2k.log"
  echo "########## ornith 32GB fine-tune mode=$MODE start $(date) ##########" | tee "$LOG"
  ~/my_env/bin/python -u -m ExistingModelFineTuning.OrnithLongContext.finetune_ornith_qlora_routed \
    --model deepreinforce-ai/Ornith-1.0-9B --corpus pg19 --attn-mode "$MODE" \
    --seq-len 2048 --quantization none \
    --val-blocks 32 --train-blocks 512 --max-steps 100 --accum 4 --loss-chunk-size 256 \
    --log-every 5 --save-every 20 --seed 1337 \
    --output-dir "checkpoints/ornith9b_${MODE}_2k" >> "$LOG" 2>&1
  echo "MODE=$MODE EXIT=$? $(date)" | tee -a "$LOG"
done
echo "=== ORNITH 32GB (seq 2048, bf16 LoRA) RUNS DONE $(date) ==="
