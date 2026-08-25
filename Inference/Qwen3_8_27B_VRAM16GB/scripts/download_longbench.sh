#!/usr/bin/env bash
# Download official LongBench-E English passage-retrieval JSONL.
# Used by tools/bench_longbench_retrieval.py as the public quality gate.
set -euo pipefail

DEST="${1:-${HGA_LONGBENCH_DIR:-$HOME/data/LongBench}}"
FILE="passage_retrieval_en_e.jsonl"
mkdir -p "$DEST"
OUT="$DEST/$FILE"

if [[ -f "$OUT" ]]; then
  echo "already have $OUT"
  wc -l "$OUT"
  exit 0
fi

python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("THUDM/LongBench", "passage_retrieval_en_e", split="test")
with out.open("w", encoding="utf-8") as handle:
    for row in ds:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
print(out)
print(f"{len(ds)} examples")
PY
