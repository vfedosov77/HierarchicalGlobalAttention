#!/usr/bin/env bash
# Ornith-1.0-9B V3-1 matched-context fine-tune (Target A: 16GB Quadro RTX 5000, Turing SM7.5).
#
# routed vs dense at seq 1536 — the largest that fits a 16GB card for the routed single-forward
# (Ornith head_dim=256 is 2x Qwen's 128, so the routed-assembly peak OOMs at 2048).  --topk-chunks
# 12 keeps the reference's 50% selection budget (12/24 chunks @1536 == 16/32 @2048).  QLoRA
# (4-bit NF4 + fp16 compute + GradScaler), single full-sequence forward, seed 1337, 32 PG19
# validation blocks.  Steady-state ~250 tok/s; each mode's 100 steps takes ~40 min.
#
# Usage (launches a detached tmux session that survives disconnects — for the overnight run):
#     ExistingModelFineTuning/OrnithLongContext/Scripts/run_ornith_16gb_1536.sh
#     tmux attach -t ornith_16gb            # watch live
#     bash .../run_ornith_16gb_1536.sh --run   # run in the foreground instead (no tmux)
set -u

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SESSION=ornith_16gb

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
  echo "  logs:   tail -f $REPO_ROOT/logs/ornith9b_routed_1536.log"
  exit 0
fi

# ---- actual work (runs inside tmux) --------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HF xet blob CDN can 403 on PG19 parquet streaming; the local shard covers the validation split
# (deterministic).  Train streams fresh from the Hub.
export HGA_LOCAL_DATA_DIR=TrainData/pg19_local
mkdir -p logs

for MODE in routed dense; do
  LOG="logs/ornith9b_${MODE}_1536.log"
  echo "########## ornith 16GB fine-tune mode=$MODE start $(date) ##########" | tee "$LOG"
  .venv/bin/python -u -m ExistingModelFineTuning.OrnithLongContext.finetune_ornith_qlora_routed \
    --model deepreinforce-ai/Ornith-1.0-9B --corpus pg19 --attn-mode "$MODE" \
    --seq-len 1536 --fp16 --quantization nf4 --topk-chunks 12 \
    --val-blocks 32 --train-blocks 512 --max-steps 100 --accum 4 --loss-chunk-size 256 \
    --log-every 5 --save-every 20 --seed 1337 \
    --output-dir "checkpoints/ornith9b_${MODE}_1536" >> "$LOG" 2>&1
  echo "MODE=$MODE EXIT=$? $(date)" | tee -a "$LOG"
done
echo "=== ORNITH 16GB (seq 1536) RUNS DONE $(date) ==="
