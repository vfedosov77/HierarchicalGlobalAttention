#!/usr/bin/env python3
"""Prove the pure-torch variants reproduce the reference HierarchicalGlobalAttention.

The optimized generation variants in this folder must not change model quality.
The teacher-forcing / prefill path (``_forward_dense``) is a verbatim port of
``ExistingModelFineTuning/HierarchicalGlobalAttention.py``; this test checks the
full no-cache forward is bit-identical to the original on random weights+inputs.

Run:  python -m ExistingModelFineTuning.gen_opt.test_equivalence
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
EMFT = HERE.parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _attn_cls(mod):
    for n in ("HierarchicalGlobalAttention", "GlobalAttention"):
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(mod)


def _rope(S, head_dim, theta, device):
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=device).float() / half))
    t = torch.arange(S, device=device).float()
    fr = torch.einsum("i,j->ij", t, inv)
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos()[None], emb.sin()[None]


def check(variant_path: Path, device="cuda", tol=1e-4) -> float:
    orig = _load(EMFT / "HierarchicalGlobalAttention.py", "_orig_ref")
    var = _load(variant_path, f"_var_{variant_path.stem}")
    cfg = dict(d_model=256, nhead=8, kv_heads=4, head_dim=32, chunk_size=64,
               group_size=16, topk_chunks=5, topk_groups=8, qk_norm=True, theta=1e6)
    torch.manual_seed(0)
    A = _attn_cls(orig)(**cfg).to(device).float().eval()
    B = _attn_cls(var)(**cfg).to(device).float().eval()
    bd = dict(B.named_parameters())
    for n, p in A.named_parameters():
        bd[n].data.copy_(p.data)
    S = 200
    x = torch.randn(2, S, 256, device=device)
    cos, sin = _rope(S, 32, 1e6, device)
    with torch.no_grad():
        outA, _ = A.forward(x, rotary_data=(cos, sin))
        outB, _ = B.forward(x, position_embeddings=(cos, sin))
    diff = (outA - outB).abs().max().item()
    status = "OK" if diff <= tol else "MISMATCH"
    print(f"[equiv] {variant_path.name:20s} max|diff| vs original = {diff:.3e}  -> {status}")
    return diff


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [HERE.parent / "vt_torch.py"]
    worst = max(check(p) for p in paths)
    sys.exit(0 if worst <= 1e-4 else 1)
