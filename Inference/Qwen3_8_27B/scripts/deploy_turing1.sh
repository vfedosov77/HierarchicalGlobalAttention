#!/usr/bin/env bash
# rsync this folder to turing1 and build there.
# Usage: HGA_SSH=vladimir@10.143.241.223 ./scripts/deploy_turing1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"  # HierarchicalGlobalAttention repo
HOST="${HGA_SSH:-vladimir@10.143.241.223}"
DEST="${HGA_REMOTE_DIR:-~/HGA}"

rsync -az \
  --exclude 'Inference/Qwen3_8_27B/build' \
  --exclude 'Inference/Qwen3_8_27B/third_party' \
  --exclude '.git' \
  "$ROOT/" "$HOST:$DEST/"

ssh "$HOST" "bash -lc 'cd $DEST/Inference/Qwen3_8_27B && chmod +x scripts/*.sh && ./scripts/setup.sh && ./build/hga-test && ./build/hga-bench --threads 40 --max-seq 32768'"
