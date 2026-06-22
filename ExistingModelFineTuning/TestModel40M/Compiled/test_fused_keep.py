#!/usr/bin/env python3
"""Evaluate the *Compiled* fused hierarchical attention as a drop-in dense
replacement, mirroring ``../test_routed_keep.py`` but driving the fused
``HierarchicalGlobalAttentionFusedExactQ.GlobalAttention`` instead of the
``HierarchicalGlobalAttentionRouted`` module.

The fused module now honours the same policy as the routed approach:

* ``attend_summaries=False`` -> only exact token K/V enter the softmax (group
  summaries are dropped), so a dense checkpoint stays loadable without
  fine-tuning the summary path;
* ``keep_first`` / ``keep_last`` -> the first/last chunks before every query
  chunk are always resident as exact tokens, and the middle range is routed.

With those two knobs the fused attention reaches the same next-token loss as the
routed module (~0.03 nats above dense on the QK fine-tuned checkpoint).  When
``attend_summaries=False`` or ``keep_first``/``keep_last`` are active the module
falls back from the Triton kernel to the exact reference path, which implements
the policy.

Examples:
    # 8K window, 2 always-resident first chunks + 6 last chunks, 5 docs:
    python test_fused_keep.py --seq-len 8192 --keep-first 2 --keep-last 6 \
        --num-batches 5

    # against a QK fine-tuned checkpoint (keys remapped self_attn.* -> attn.*):
    python test_fused_keep.py --seq-len 8192 --keep-first 2 --keep-last 6 \
        --ha-checkpoint ../speed_run_ha_from_dense_adamw_kq_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import benchmark_no_summaries_loss as B
from benchmark_no_summaries_loss import (
    DEVICE,
    DEFAULT_CKPT,
    CHUNK_SIZE,
    GROUP_SIZE,
    HIDDEN_DIM,
    KV_HEADS,
    NUM_HEADS,
    NUM_LAYERS,
    TOPK_CHUNKS,
    TOPK_GROUPS,
    AttnWrapper,
    SmallLM,
    build_model,
    eval_loss,
    load_checkpoint,
)

from HierarchicalGlobalAttentionFusedExactQ import GlobalAttention as FusedGlobalAttention


def build_fused_model(keep_first: int, keep_last: int,
                      topk_chunks: int, topk_groups: int,
                      attend_summaries: bool) -> SmallLM:
    """SmallLM whose attention is the Compiled fused ``GlobalAttention`` run in
    token-only mode with the ``keep_first`` / ``keep_last`` window."""

    def attn_factory(layer_idx: int):
        attn = FusedGlobalAttention(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            kv_heads=KV_HEADS,
            dropout=0.0,
            use_bias_q=False,
            use_bias_k=False,
            use_bias_v=False,
            use_bias_o=False,
            causal=True,
            use_global=True,
            chunk_size=CHUNK_SIZE,
            group_size=GROUP_SIZE,
            topk_chunks=topk_chunks,
            topk_groups=topk_groups,
            return_router_stats=False,
            head_dim=HIDDEN_DIM // NUM_HEADS,
            qk_norm=False,
            attend_summaries=attend_summaries,
            keep_first=keep_first,
            keep_last=keep_last,
        )
        return AttnWrapper(attn)

    return SmallLM(attn_factory)


def get_fineweb_batches(seq_len: int, doc_index: int, num_batches: int):
    """Return up to ``num_batches`` sequences of ``seq_len + 1`` GPT-2 tokens,
    each from a distinct FineWeb doc with enough tokens (matches
    ``../test_routed_keep.py`` so the two scripts evaluate identical data)."""
    import pandas as pd
    from transformers import GPT2TokenizerFast

    candidates = [
        os.path.join(SCRIPT_DIR, "..", "NotOptimized", "fineweb_sample",
                     "sample", "10BT", "000_00000.parquet"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "fineweb_sample",
                     "sample", "10BT", "000_00000.parquet"),
    ]
    parquet = next((p for p in candidates if os.path.exists(p)), None)
    if parquet is None:
        raise FileNotFoundError(f"No FineWeb parquet found. Tried: {candidates}")
    print(f"[data] reading {os.path.abspath(parquet)}")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    df = pd.read_parquet(parquet, columns=["text"])
    need = seq_len + 1
    seqs, used, skipped = [], [], 0
    i = doc_index
    while len(seqs) < num_batches and i < len(df):
        ids = tok(df["text"].iloc[i], add_special_tokens=False)["input_ids"]
        if len(ids) >= need:
            seqs.append(torch.tensor(ids[:need], dtype=torch.long))
            used.append(i)
        else:
            skipped += 1
        i += 1
    if len(seqs) < num_batches:
        print(f"[warn] only found {len(seqs)}/{num_batches} docs with >= {need} tokens")
    print(f"[data] using {len(seqs)} docs (#{used[0]}..#{used[-1]}), "
          f"skipped {skipped} short docs")
    return seqs


@torch.no_grad()
def eval_mean_loss(model: SmallLM, seqs) -> float:
    total = 0.0
    for seq in seqs:
        inputs = seq[:-1].unsqueeze(0).to(DEVICE)
        targets = seq[1:].unsqueeze(0).to(DEVICE)
        loss, _ = eval_loss(model, inputs, targets)
        total += loss
    return total / len(seqs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,
                        help="Dense weights (also fused weights if --ha-checkpoint unset).")
    parser.add_argument("--ha-checkpoint", default=None,
                        help="Optional QK fine-tuned checkpoint for the fused model "
                             "(keys remapped self_attn.* -> self_attn.attn.*).")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--doc-index", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=5,
                        help="Number of distinct FineWeb docs to average the loss over.")
    parser.add_argument("--keep-first", type=int, default=2)
    parser.add_argument("--keep-last", type=int, default=6)
    parser.add_argument("--topk-chunks", type=int, default=TOPK_CHUNKS)
    parser.add_argument("--topk-groups", type=int, default=TOPK_GROUPS)
    parser.add_argument("--with-summaries", action="store_true",
                        help="Keep the group summaries in the softmax (debug). "
                             "Default drops them (token-only) to match the routed "
                             "dense-checkpoint setup.")
    args = parser.parse_args()

    if DEVICE != "cuda":
        print("[warn] CUDA not available; the fused path falls back to the "
              "reference and runs on CPU (slow).")

    torch.manual_seed(0)
    seqs = get_fineweb_batches(args.seq_len, args.doc_index, args.num_batches)
    print(f"[data] seq_len={args.seq_len} x {len(seqs)} batches on {DEVICE}")
    print(f"[cfg] keep_first={args.keep_first} keep_last={args.keep_last} "
          f"topk_chunks={args.topk_chunks} topk_groups={args.topk_groups} "
          f"(chunk={CHUNK_SIZE} group={GROUP_SIZE})")
    print(f"[cfg] attend_summaries={args.with_summaries} "
          f"(token-only={not args.with_summaries})\n")

    routed_ckpt = args.ha_checkpoint or args.checkpoint
    routed_remap = args.ha_checkpoint is not None

    # ---- dense baseline -----------------------------------------------------
    dense = build_model(use_global=False, attend_summaries=True,
                        topk_chunks=args.topk_chunks,
                        topk_groups=args.topk_groups).to(DEVICE).float()
    load_checkpoint(dense, args.checkpoint)
    dense_loss = eval_mean_loss(dense, seqs)
    dense_ppl = float(torch.exp(torch.tensor(dense_loss)))
    print(f"[{'dense':>14}] loss={dense_loss:.5f}  ppl={dense_ppl:8.3f}")
    del dense
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ---- fused hierarchical (token-only + keep_first / keep_last) -----------
    fused = build_fused_model(keep_first=args.keep_first,
                              keep_last=args.keep_last,
                              topk_chunks=args.topk_chunks,
                              topk_groups=args.topk_groups,
                              attend_summaries=args.with_summaries).to(DEVICE).float()
    load_checkpoint(fused, routed_ckpt, remap_self_attn=routed_remap)
    path = fused.layers[0].self_attn.attn._last_path \
        if hasattr(fused.layers[0].self_attn.attn, "_last_path") else "?"
    fused_loss = eval_mean_loss(fused, seqs)
    fused_ppl = float(torch.exp(torch.tensor(fused_loss)))
    try:
        path = fused.layers[0].self_attn.attn._last_path
    except Exception:
        path = "?"
    print(f"[{'fused':>14}] loss={fused_loss:.5f}  ppl={fused_ppl:8.3f}  "
          f"attn_path={path}")

    delta = fused_loss - dense_loss
    print("\n================ summary ================")
    print(f"  dense  : loss={dense_loss:.5f}  ppl={dense_ppl:8.3f}")
    print(f"  fused  : loss={fused_loss:.5f}  ppl={fused_ppl:8.3f}")
    print(f"  Δloss (fused - dense) = {delta:+.5f} nats "
          f"({(fused_ppl / dense_ppl - 1.0) * 100:+.1f}% ppl)")


if __name__ == "__main__":
    main()
