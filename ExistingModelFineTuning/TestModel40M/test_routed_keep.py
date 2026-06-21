#!/usr/bin/env python3
"""Tune ``HierarchicalGlobalAttentionRouted`` against dense on one FineWeb seq.

Drives the routed attention **directly** (sparse chunk-routing path via a fresh
KV cache) with adjustable ``keep_first`` / ``keep_last`` (always-resident first/
last chunks) and the top-k budget, then prints its next-token loss next to the
native dense loss so you can play with the routing policy.

Defaults: keep_first=6, keep_last=2.

Examples:
    # routing approximation only (same weights for dense and routed, no HA fine-tune):
    python ExistingModelFineTuning/TestModel40M/test_routed_keep.py \
        --seq-len 8192 --keep-first 6 --keep-last 6 --num-batches 20

    # against the QK fine-tuned checkpoint:
    python ExistingModelFineTuning/TestModel40M/test_routed_keep.py \
        --seq-len 8192 --keep-first 4 --keep-last 6 --num-batches 20 \
        --ha-checkpoint ExistingModelFineTuning/TestModel40M/speed_run_ha_from_dense_adamw_kq_final.pt

Notes:
* dense always loads ``--checkpoint`` (original dense weights).
* the routed model loads ``--ha-checkpoint`` if given (QK fine-tuned, keys
  remapped), otherwise ``--checkpoint``.
* the routed attention runs ``attend(use_summaries=False)`` — only exact token
  K/V enter the softmax, so it stays loadable from a dense checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# benchmark_no_summaries_loss lives in the Compiled/ sibling folder.
COMPILED_DIR = os.path.join(SCRIPT_DIR, "Compiled")
for _p in (SCRIPT_DIR, COMPILED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    SmallLM,
    RoutedWrapper,
    build_model,
    eval_loss,
    get_fineweb_sequence,
    load_checkpoint,
)


class IncrementalRoutedWrapper(nn.Module):
    """Streams the sequence **chunk-by-chunk** through a shared KV cache so the
    routed module takes its incremental ``decode_block`` path.

    Both the vectorized prefill (``assemble_routed_kv``) and the incremental
    ``decode_block`` now honour the ``ChunkPlacementPolicy`` (``keep_first`` /
    ``keep_last``), so the two paths produce comparable losses.  Feeding one
    chunk at a time keeps ``start_pos > 0`` after the first chunk, routing every
    block through ``decode_block`` — use ``--incremental`` to cross-check that
    the incremental path agrees with the default vectorized prefill.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def _get_rotary(self, seq_len: int, device: torch.device):
        head_dim = self.attn.head_dim
        theta = self.attn.theta
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from transformers import DynamicCache
        B, S, _ = x.shape
        device = x.device
        C = self.attn.chunk_size
        cos, sin = self._get_rotary(S, device)
        cache = DynamicCache()
        outs = []
        p = 0
        while p < S:
            take = min(C - (p % C), S - p)
            xs = x[:, p:p + take]
            pe = (cos[p:p + take].unsqueeze(0).expand(B, -1, -1),
                  sin[p:p + take].unsqueeze(0).expand(B, -1, -1))
            pos = torch.arange(p, p + take, device=device)
            out, _ = self.attn(hidden_states=xs, position_embeddings=pe,
                               past_key_value=cache, cache_position=pos)
            outs.append(out)
            p += take
        return torch.cat(outs, dim=1)


def build_routed_model(use_cache: bool, keep_first: int, keep_last: int,
                       topk_chunks: int, topk_groups: int,
                       incremental: bool = False) -> SmallLM:
    if B.HierarchicalGlobalAttentionRouted is None:
        raise RuntimeError("HierarchicalGlobalAttentionRouted unavailable "
                           "(KvRouter import failed).")

    def attn_factory(layer_idx: int):
        attn = B.HierarchicalGlobalAttentionRouted(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            kv_heads=KV_HEADS,
            dropout=0.0,
            use_bias_q=False,
            use_bias_k=False,
            use_bias_v=False,
            use_bias_o=False,
            causal=True,
            chunk_size=CHUNK_SIZE,
            group_size=GROUP_SIZE,
            topk_chunks=topk_chunks,
            topk_groups=topk_groups,
            keep_first=keep_first,
            keep_last=keep_last,
            head_dim=HIDDEN_DIM // NUM_HEADS,
            qk_norm=False,
            layer_idx=layer_idx,
            cache_location="vram",
            num_layers=NUM_LAYERS,
        )
        if incremental:
            return IncrementalRoutedWrapper(attn)
        return RoutedWrapper(attn, use_cache=use_cache)

    return SmallLM(attn_factory)


def get_fineweb_batches(seq_len: int, doc_index: int, num_batches: int):
    """Return up to ``num_batches`` sequences of ``seq_len + 1`` GPT-2 tokens,
    each taken from a distinct FineWeb doc with enough tokens."""
    import pandas as pd
    from transformers import GPT2TokenizerFast

    candidates = [
        os.path.join(SCRIPT_DIR, "NotOptimized", "fineweb_sample",
                     "sample", "10BT", "000_00000.parquet"),
        os.path.join(SCRIPT_DIR, "..", "..", "fineweb_sample",
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
    """Mean next-token loss over a list of equal-length sequences."""
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
                        help="Dense weights (also routed weights if --ha-checkpoint unset).")
    parser.add_argument("--ha-checkpoint", default=None,
                        help="Optional QK fine-tuned checkpoint for the routed model "
                             "(keys remapped self_attn.* -> self_attn.attn.*).")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--doc-index", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=20,
                        help="Number of distinct FineWeb docs to average the loss over.")
    parser.add_argument("--keep-first", type=int, default=6)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--topk-chunks", type=int, default=TOPK_CHUNKS)
    parser.add_argument("--topk-groups", type=int, default=TOPK_GROUPS)
    parser.add_argument("--no-cache", action="store_true",
                        help="Drive the routed module with past_key_value=None "
                             "instead of a fresh DynamicCache.")
    parser.add_argument("--incremental", action="store_true",
                        help="Stream the sequence chunk-by-chunk through the "
                             "decode_block path instead of the vectorized "
                             "prefill. Both honour keep_first/keep_last; use this "
                             "to cross-check the two paths agree.")
    args = parser.parse_args()

    if DEVICE != "cuda":
        print("[warn] CUDA not available; routing runs on CPU and will be slow.")

    torch.manual_seed(0)
    seqs = get_fineweb_batches(args.seq_len, args.doc_index, args.num_batches)
    print(f"[data] seq_len={args.seq_len} x {len(seqs)} batches on {DEVICE}")
    print(f"[cfg] keep_first={args.keep_first} keep_last={args.keep_last} "
          f"topk_chunks={args.topk_chunks} topk_groups={args.topk_groups} "
          f"(chunk={CHUNK_SIZE} group={GROUP_SIZE})")
    if args.incremental:
        print("[cfg] routing path = incremental decode_block "
              "(keep_first/keep_last active)\n")
    else:
        print("[cfg] routing path = vectorized prefill "
              "(keep_first/keep_last active)\n")

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

    # ---- routed (keep_first / keep_last) ------------------------------------
    routed = build_routed_model(use_cache=not args.no_cache,
                                keep_first=args.keep_first,
                                keep_last=args.keep_last,
                                topk_chunks=args.topk_chunks,
                                topk_groups=args.topk_groups,
                                incremental=args.incremental).to(DEVICE).float()
    load_checkpoint(routed, routed_ckpt, remap_self_attn=routed_remap)
    routed_loss = eval_mean_loss(routed, seqs)
    routed_ppl = float(torch.exp(torch.tensor(routed_loss)))
    print(f"[{'routed':>14}] loss={routed_loss:.5f}  ppl={routed_ppl:8.3f}")

    delta = routed_loss - dense_loss
    print("\n================ summary ================")
    print(f"  dense  : loss={dense_loss:.5f}  ppl={dense_ppl:8.3f}")
    print(f"  routed : loss={routed_loss:.5f}  ppl={routed_ppl:8.3f}")
    print(f"  Δloss (routed - dense) = {delta:+.5f} nats "
          f"({(routed_ppl / dense_ppl - 1.0) * 100:+.1f}% ppl)")


if __name__ == "__main__":
    main()
