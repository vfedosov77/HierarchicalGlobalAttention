#!/usr/bin/env bash
# Download Unsloth Q4_K_M GGUF for Qwen3.8-27B (best AVX-512 INT4 on Skylake-SP).
# IQ4_* quants want VNNI, which Xeon Gold 6148 does not have.
set -euo pipefail

DEST="${1:-${HGA_MODEL_DIR:-$HOME/models/Qwen3.8-27B-GGUF}}"
REPO="${HGA_HF_REPO:-unsloth/Qwen3.8-27B-GGUF}"
# Prefer Q4_K_M; UD-Q4_K_XL is a bit larger/better if you have the RAM.
FILE="${HGA_GGUF_FILE:-Qwen3.8-27B-Q4_K_M.gguf}"

mkdir -p "$DEST"
cd "$DEST"

if [[ -f "$FILE" ]]; then
  echo "already have $DEST/$FILE"
  ls -lh "$FILE"
  exit 0
fi

# turing1 has no internet. Copy a GGUF from a machine that can reach Hugging Face:
#   scp Qwen3.8-27B-Q4_K_M.gguf vladimir@turing1:~/models/Qwen3.8-27B-GGUF/

if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --include "$FILE" --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" "$FILE" --local-dir "$DEST"
else
  python3 -m pip install --user -q "huggingface_hub[cli]"
  python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$REPO', '$FILE', local_dir='$DEST'))"
fi

ls -lh "$DEST"
echo "export HGA_MODEL=$DEST/$FILE"
