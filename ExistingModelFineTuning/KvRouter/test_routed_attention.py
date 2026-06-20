"""Equivalence + smoke tests for HierarchicalGlobalAttentionRouted.

  1. new stateless forward  ==  old `_forward_dense`  (vectorized.py is a port of it)
  2. generation prefill+decode runs, finite, right shapes, store stays bounded.

Run:  python -m ExistingModelFineTuning.KvRouter.test_routed_attention
"""

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EFT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(EFT_DIR)
for p in (ROOT_DIR, EFT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from ExistingModelFineTuning.HierarchicalGlobalAttention import HierarchicalGlobalAttention
from ExistingModelFineTuning.HierarchicalGlobalAttentionRouted import HierarchicalGlobalAttentionRouted


def _rotary(S, Dh, theta, device):
    half = Dh // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(S, device=device, dtype=torch.float32)
    emb = torch.cat([torch.outer(pos, inv)] * 2, dim=-1)
    return emb.cos(), emb.sin()  # [S, Dh] each


def test_equivalence(device):
    torch.manual_seed(0)
    cfg = dict(d_model=64, nhead=4, kv_heads=2, head_dim=16, chunk_size=8, group_size=4,
               topk_chunks=3, topk_groups=4, theta=10000.0, qk_norm=False,
               use_bias_q=False, use_bias_k=False, use_bias_v=False, use_bias_o=False)
    old = HierarchicalGlobalAttention(**cfg).to(device).eval()
    new = HierarchicalGlobalAttentionRouted(**cfg).to(device).eval()
    new.load_state_dict(old.state_dict(), strict=True)

    B, S = 2, 37
    x = torch.randn(B, S, cfg["d_model"], device=device)
    cos, sin = _rotary(S, cfg["head_dim"], cfg["theta"], device)

    with torch.no_grad():
        out_old, _ = old(x, rotary_data=(cos, sin))
        out_new, _ = new(hidden_states=x, position_embeddings=(cos, sin))

    err = (out_old - out_new).abs().max().item()
    print(f"[equivalence] new-stateless vs old-dense max abs err = {err:.3e}")
    assert err < 1e-4, err


def test_generation(device):
    torch.manual_seed(1)
    cfg = dict(d_model=64, nhead=4, kv_heads=2, head_dim=16, chunk_size=8, group_size=4,
               topk_chunks=3, topk_groups=4, theta=10000.0, qk_norm=False,
               use_bias_q=False, use_bias_k=False, use_bias_v=False, use_bias_o=False,
               keep_last=1, keep_first=1)

    class _Cache:  # stand-in for HF DynamicCache; the router attaches itself to it
        pass

    for location in ("vram", "ram"):
        new = HierarchicalGlobalAttentionRouted(cache_location=location, **cfg).to(device).eval()
        B, ctx, gen = 1, 20, 12
        x = torch.randn(B, ctx + gen, cfg["d_model"], device=device)
        cos, sin = _rotary(ctx + gen, cfg["head_dim"], cfg["theta"], device)
        cache = _Cache()
        with torch.no_grad():
            # prefill
            cp = torch.arange(ctx, device=device)
            out, _ = new(hidden_states=x[:, :ctx], position_embeddings=(cos[:ctx], sin[:ctx]),
                         past_key_value=cache, cache_position=cp, use_cache=True)
            assert out.shape == (B, ctx, cfg["d_model"]) and torch.isfinite(out).all()
            # decode token by token
            for i in range(gen):
                p = ctx + i
                cp = torch.tensor([p], device=device)
                step, _ = new(hidden_states=x[:, p:p + 1],
                              position_embeddings=(cos[p:p + 1], sin[p:p + 1]),
                              past_key_value=cache, cache_position=cp, use_cache=True)
                assert step.shape == (B, 1, cfg["d_model"]) and torch.isfinite(step).all()
        router = cache._hga_router
        print(f"[generation:{location}] ok, closed chunks = {router.store.num_closed_chunks(0)}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    test_equivalence(dev)
    test_generation(dev)
    print("ALL PASSED")
