#!/usr/bin/env python3
"""Download model checkpoints from Hugging Face into ExistingModelFineTuning/.

Usage:
    python prepare_model.py              # download both checkpoints
    python prepare_model.py --dense-only # download only the dense baseline
    python prepare_model.py --force      # re-download even if files already exist
"""

import argparse
import os
import sys

REPO_ID = "vfedosov/HierarchicalGlobalAttention"
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ExistingModelFineTuning")

CHECKPOINTS = {
    "dense": "speed_run_dense_muon_final.pt",
    "ha":    "speed_run_ha_from_dense_adamw_kq_final.pt",
}


def download(filename: str, dest_dir: str, force: bool) -> str:
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and not force:
        print(f"  already exists, skipping: {dest}")
        return dest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")

    print(f"  downloading {filename} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
        local_dir=dest_dir,
    )
    print(f"  saved to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HA model checkpoints from Hugging Face")
    parser.add_argument("--dense-only", action="store_true", help="Download only the dense baseline checkpoint")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists")
    args = parser.parse_args()

    os.makedirs(DEST_DIR, exist_ok=True)

    targets = ["dense"] if args.dense_only else list(CHECKPOINTS)
    for key in targets:
        filename = CHECKPOINTS[key]
        print(f"[{key}] {filename}")
        download(filename, DEST_DIR, force=args.force)

    print("\nAll done. Checkpoints are in:", DEST_DIR)


if __name__ == "__main__":
    main()
