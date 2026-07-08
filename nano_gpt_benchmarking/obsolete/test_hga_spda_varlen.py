"""Standalone test / mini-training for chunk-routed varlen attention.

Two things:
  1. Correctness checks (synthetic, fast): document isolation + token-level value causality
     of ``chunk_routed_varlen_attention``.
  2. A tiny 1-layer transformer trained on a single ~50K-token packed sequence built from
     many FineWeb documents (tokenized with the workspace GPT-2 tokenizer). Reports loss and
     tokens/s so the routing path can be exercised end-to-end and its efficiency observed.

Runs on any CUDA GPU: the FA3 routing path is Hopper-only, so this uses the numerically
equivalent pure-torch router (default). Set ROUTE_FLASH=1 on Hopper to exercise FA3.

Usage:
    python test_hga_spda_varlen.py                 # correctness checks + 50K training
    python test_hga_spda_varlen.py --check-only    # just the correctness checks
    python test_hga_spda_varlen.py --tokens 50000 --steps 30
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hga_spda_attention import chunk_routed_varlen_attention

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "fineweb_sample", "sample", "10BT", "000_00000.parquet")
TOKENIZER = os.path.join(ROOT, "gpt2_tokenizer", "tokenizer.json")
BOS_ID = 50256  # <|endoftext|>, used as document separator (matches the speedrun loader)


# ----------------------------------------------------------------------------------------
# Rotary (absolute positions; cross-document attention is masked out anyway, and documents
# are contiguous, so absolute-position RoPE gives correct in-document relative phases).
# ----------------------------------------------------------------------------------------
def build_rotary(seq_len: int, head_dim: int, device, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # [T, half]
    return torch.cos(freqs), torch.sin(freqs)             # each [T, half]


def apply_rotary(x, cos, sin):
    # x: [T, H, D]
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _qk_norm(x):
    return F.rms_norm(x, (x.size(-1),))


# ----------------------------------------------------------------------------------------
# Correctness checks
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def check_doc_isolation_and_causality(device):
    torch.manual_seed(0)
    H, D, CH = 3, 32, 8
    # Three documents of uneven length (not multiples of CH), packed into one sequence.
    lengths = [20, 13, 27]
    T = sum(lengths)
    cu = torch.tensor([0, 20, 33, 60], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(T, H, D, device=device)
    k = torch.randn(T, H, D, device=device)
    v = torch.randn(T, H, D, device=device)

    y0 = chunk_routed_varlen_attention(q, k, v, cu, chunk_size=CH, topk=4,
                                       softmax_scale=scale, detach_routed=False)

    # (a) Document isolation: perturb doc #2 (tokens 20..32); docs #1 and #3 must be identical.
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    k2[20:33] += 5.0
    v2[20:33] += 5.0
    q2[20:33] += 5.0
    y1 = chunk_routed_varlen_attention(q2, k2, v2, cu, chunk_size=CH, topk=4,
                                       softmax_scale=scale, detach_routed=False)
    d1 = (y1[:20] - y0[:20]).abs().max().item()
    d3 = (y1[33:] - y0[33:]).abs().max().item()
    assert d1 < 1e-5, f"doc-1 leaked from doc-2 perturbation: {d1}"
    assert d3 < 1e-5, f"doc-3 leaked from doc-2 perturbation: {d3}"

    # (b) Value causality: perturbing a token strictly in the FUTURE (and in a later chunk,
    # so routing selection cannot legitimately depend on it) must not change earlier outputs.
    # Token 0 lives in chunk 0 of doc-1; perturb token 15 (chunk 1 of doc-1, strictly later).
    qf, kf, vf = q.clone(), k.clone(), v.clone()
    kf[15] += 5.0
    vf[15] += 5.0
    qf[15] += 5.0
    y2 = chunk_routed_varlen_attention(qf, kf, vf, cu, chunk_size=CH, topk=4,
                                       softmax_scale=scale, detach_routed=False)
    # tokens 0..7 (chunk 0) must be unaffected by token 15 (a strictly-later chunk)
    dcaus = (y2[:8] - y0[:8]).abs().max().item()
    assert dcaus < 1e-5, f"value causality violated: earlier tokens changed by a later token: {dcaus}"

    print(f"[ok] doc isolation (max leak {max(d1, d3):.2e}) and value causality (max {dcaus:.2e})")


def check_onehot_route_decode(device):
    """Validate the FA3 one-hot-positional-V decode (via its torch simulation).

    The one-hot output slot magnitudes must recover the exact per-chunk attention weights,
    so the top-k they yield must match a direct windowed top-k over the chunk scores.
    """
    from hga_spda_attention import (
        _doc_chunk_layout, _onehot_out_torch, _decode_onehot_route,
    )

    torch.manual_seed(1)
    H, D, CH = 4, 64, 8
    lengths = [200, 130, 270]                       # docs many-chunks long (chunks per doc up to ~34)
    T = sum(lengths)
    cu = torch.tensor([0, 200, 330, 600], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(D)
    topk, window = 5, D - 1

    L = _doc_chunk_layout(cu, T, CH, device)
    C = L["C"]
    q_sum = torch.randn(C, H, D, device=device)
    k_sum = torch.randn(C, H, D, device=device)

    # Decode via the one-hot simulation.
    out = _onehot_out_torch(q_sum, k_sum, L, scale, window)
    sel, slot_valid = _decode_onehot_route(out, L, topk, window)

    # Reference: the true routed weight is mean-over-heads of the windowed causal softmax.
    # The decode must recover exactly this, so its top-k (strictly-past) must match.
    scores_h = torch.einsum("ihd,jhd->hij", q_sum, k_sum) * scale     # [H, C, C]
    ii = torch.arange(C, device=device)
    same_doc = L["doc_of_chunk"][:, None] == L["doc_of_chunk"][None, :]
    causal = ii[None, :] <= ii[:, None]                              # softmax includes the diagonal
    within = (ii[:, None] - ii[None, :]) <= window
    scores_h = scores_h.masked_fill(~(same_doc & causal & within)[None], float("-inf"))
    attn = torch.softmax(scores_h, dim=-1).mean(0)                    # [C, C] mean-head weights
    past = ii[None, :] < ii[:, None]                                  # selection is strictly past
    ref = attn.masked_fill(~(same_doc & past & within), float("-inf"))
    ref_topv, ref_topi = ref.topk(topk, dim=-1)
    ref_valid = ref_topv > -1e30

    # Compare selected chunk SETS per query (order can differ on ties; sets must match).
    mismatch = 0
    for c in range(C):
        got = set(sel[c, :-1][slot_valid[c, :-1]].tolist())          # drop own-chunk slot
        exp = set(ref_topi[c][ref_valid[c]].tolist())
        if got != exp:
            mismatch += 1
    assert mismatch == 0, f"one-hot decode disagreed with direct top-k on {mismatch}/{C} query chunks"
    print(f"[ok] one-hot-V route decode matches direct windowed top-k on all {C} chunks "
          f"(H={H}, head_dim={D}, window={window}, topk={topk})")


# ----------------------------------------------------------------------------------------
# Data: pack FineWeb documents into one ~N-token sequence
# ----------------------------------------------------------------------------------------
def load_packed_sequence(num_tokens: int, chunk_size: int):
    import pyarrow.parquet as pq
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER)
    pf = pq.ParquetFile(PARQUET)

    ids = []
    starts = [0]
    for batch in pf.iter_batches(batch_size=64, columns=["text"]):
        for text in batch.column("text").to_pylist():
            doc = [BOS_ID] + tok(text)["input_ids"]
            ids.extend(doc)
            starts.append(len(ids))
            if len(ids) >= num_tokens + 1:
                break
        if len(ids) >= num_tokens + 1:
            break

    # Truncate to a whole number of chunks (>=16) and rebuild boundaries.
    T = (min(len(ids) - 1, num_tokens) // chunk_size) * chunk_size
    ids = ids[: T + 1]
    boundaries = [s for s in starts if s < T] + [T]
    boundaries = sorted(set(boundaries))
    input_ids = torch.tensor(ids[:T], dtype=torch.long)
    targets = torch.tensor(ids[1: T + 1], dtype=torch.long)
    cu = torch.tensor(boundaries, dtype=torch.int32)
    return input_ids, targets, cu


# ----------------------------------------------------------------------------------------
# Tiny 1-layer transformer using the routed attention
# ----------------------------------------------------------------------------------------
class TinyRoutedLM(nn.Module):
    def __init__(self, vocab: int, dim: int, num_heads: int, head_dim: int,
                 chunk_size: int, topk: int):
        super().__init__()
        assert dim == num_heads * head_dim
        self.num_heads, self.head_dim, self.dim = num_heads, head_dim, dim
        self.chunk_size, self.topk = chunk_size, topk
        self.scale = 1.0 / math.sqrt(head_dim)
        # Route via the FA3 one-hot-V simulation when requested (testable off Hopper).
        self.use_onehot_sim = os.environ.get("ROUTE_ONEHOT_SIM", "0") != "0"

        self.embed = nn.Embedding(vocab, dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tie
        nn.init.normal_(self.embed.weight, std=0.02)  # small tied init -> logits ~O(1), loss ~ln(vocab)

    def forward(self, input_ids, cu_seqlens, cos, sin):
        H, Dh = self.num_heads, self.head_dim
        T = input_ids.size(0)
        x = self.embed(input_ids)                                    # [T, dim]

        h = F.rms_norm(x, (self.dim,))
        q, k, v = self.qkv(h).view(T, 3, H, Dh).unbind(1)           # each [T, H, Dh]
        q, k = _qk_norm(q), _qk_norm(k)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        y = chunk_routed_varlen_attention(
            q, k, v, cu_seqlens,
            chunk_size=self.chunk_size, topk=self.topk,
            softmax_scale=self.scale, detach_routed=True,
            use_onehot_sim=self.use_onehot_sim,
        )                                                            # [T, H, Dh]
        x = x + self.o(y.reshape(T, self.dim))

        h = F.rms_norm(x, (self.dim,))
        x = x + self.fc2(F.relu(self.fc1(h)) ** 2)

        return F.rms_norm(x, (self.dim,))  # final hidden; lm_head applied block-wise in the loss

    def loss_backward(self, hidden, targets, block: int = 4096):
        # Chunked lm-head: backprop each token block into a *detached* hidden so only one
        # [block, vocab] logits tensor is alive at a time, then a single backward through the
        # model. Avoids materializing/retaining the full [T, vocab] graph (huge with vocab~50k).
        T = hidden.size(0)
        hd = hidden.detach().requires_grad_(True)
        total = 0.0
        for s in range(0, T, block):
            e = min(s + block, T)
            logits = self.lm_head(hd[s:e]).float()
            l = F.cross_entropy(logits, targets[s:e], reduction="sum") / T
            l.backward()
            total += l.item()
        hidden.backward(hd.grad)
        return total


def train(device, num_tokens, steps, chunk_size, topk):
    torch.manual_seed(0)
    dim, num_heads, head_dim = 384, 6, 64
    vocab = 50304  # next multiple of 128 >= 50257

    print(f"Loading FineWeb, packing ~{num_tokens} tokens (chunk={chunk_size}) ...")
    input_ids, targets, cu = load_packed_sequence(num_tokens, chunk_size)
    T = input_ids.size(0)
    ndocs = cu.numel() - 1
    lens = (cu[1:] - cu[:-1]).float()
    print(f"packed T={T} tokens across {ndocs} docs (mean {lens.mean():.0f}, "
          f"min {int(lens.min())}, max {int(lens.max())} tok/doc), n_chunks~{T // chunk_size}")

    input_ids = input_ids.to(device)
    targets = targets.to(device)
    cu = cu.to(device)
    cos, sin = build_rotary(T, head_dim, device)

    model = TinyRoutedLM(vocab, dim, num_heads, head_dim, chunk_size, topk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)

    model.train()
    t_step = None
    for step in range(steps):
        if step == 1:
            torch.cuda.synchronize()
            t_step = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(input_ids, cu, cos, sin)
        loss = model.loss_backward(hidden, targets)
        opt.step()
        if step == 0 or (step + 1) % 5 == 0:
            print(f"step {step:3d}  loss {loss:.4f}")

    torch.cuda.synchronize()
    if t_step is not None and steps > 1:
        dt = (time.time() - t_step) / (steps - 1)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[perf] {dt * 1e3:.1f} ms/step  {T / dt:,.0f} tok/s  peak {peak:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_doc_isolation_and_causality(device)
    check_onehot_route_decode(device)
    if args.check_only:
        return
    train(device, args.tokens, args.steps, args.chunk_size, args.topk)


if __name__ == "__main__":
    main()
