#!/usr/bin/env bash
# Run the HGA FastInference validation + decode benchmark on the RTX 5090 server.
#
# Usage:
#   bash FastInference/scripts/run_on_5090.sh [VENV_ACTIVATE]
#
# Assumes the repo is checked out and a CUDA-enabled PyTorch (>=2.4) venv exists.
# Pass the path to the venv's activate script as $1, or edit VENV below.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV="${1:-${VENV:-}}"
if [[ -n "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV"
fi

echo "=== Python / Torch / GPU ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PY

echo
echo "=== 1. Correctness: fused kernel + dense-equivalence ==="
python -m FastInference.tests.test_fused_kernels cuda

echo
echo "=== 2. Decode benchmark — full Qwen3-30B attention shapes, 32K ctx, 1024 out ==="
# 48 layers (Qwen3-30B-A3B), 32K context, 1024 decode tokens.
python -m FastInference.bench.bench_decode_5090 \
  --layers 48 --context 32768 --decode 1024 \
  --q-heads 32 --kv-heads 4 --head-dim 128 \
  --chunk 64 --group 16 --page 16 \
  --keep-first 2 --keep-last 8 \
  --topk-chunks 16 --topk-groups 64 \
  --gpu-token-chunks 512

echo
echo "=== 3. Page-size tradeoff sweep (16 vs 64) ==="
for PG in 16 64; do
  echo "--- page_size=$PG ---"
  python -m FastInference.bench.bench_decode_5090 \
    --layers 48 --context 32768 --decode 256 \
    --page "$PG" --topk-chunks 16 --topk-groups 64
done

echo
echo "Done. Compare against acceptance: decode>=15 tok/s, p50<=125ms, p99<=180ms."
