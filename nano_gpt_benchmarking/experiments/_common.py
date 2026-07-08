"""Shared helpers for the routing experiments.

These scripts live OUTSIDE the production module on purpose: they import
``hga_spda_attention`` from the parent dir and poke at its internals.
"""
import os
import sys

import torch
import torch.nn.functional as F

# The routers build their chunk summaries in fp32 (``q.view(...).float().mean(...)``), so the
# routing einsum is an fp32 matmul. Enable TF32 tensor cores for it -- this is exactly what the
# "TensorFloat32 ... not enabled" UserWarning recommends, and routing is only a top-k SELECTION
# so the reduced mantissa never changes which chunks are picked.
torch.set_float32_matmul_precision("high")

# Import the production module from the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hga_spda_attention as M  # noqa: E402


def make_packed_batch(T, H, D, n_docs, max_num_docs, device, dtype=torch.bfloat16, seed=0):
    """Build a fully-packed [T,H,D] q/k/v and a fixed-length (padded) cu_seqlens.

    Mirrors the training loader: ``cu_seqlens`` always has ``max_num_docs`` entries;
    trailing padding entries equal T. ``n_docs`` real documents partition [0, T).
    q/k are RMS-normed per head (the model applies norm(x) to q,k before routing).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, H, D, generator=g).to(device=device, dtype=dtype)
    k = torch.randn(T, H, D, generator=g).to(device=device, dtype=dtype)
    v = torch.randn(T, H, D, generator=g).to(device=device, dtype=dtype)
    # Match the model: RMS-norm q,k per head before routing.
    q = F.rms_norm(q.float(), (D,)).to(dtype)
    k = F.rms_norm(k.float(), (D,)).to(dtype)

    # Random document boundaries: n_docs segments, each a multiple-ish of chunk_size.
    # We just pick sorted cut points in (0, T).
    if n_docs <= 1:
        cuts = []
    else:
        cuts = sorted(
            torch.randperm(T - 1, generator=g)[: n_docs - 1].add(1).tolist()
        )
    ends = cuts + [T]
    cu = torch.full((max_num_docs,), T, dtype=torch.int32)
    cu[0] = 0
    # cumulative doc ends go into positions 1..n_docs
    for i, e in enumerate(ends):
        cu[1 + i] = e
    return q, k, v, cu.to(device)
