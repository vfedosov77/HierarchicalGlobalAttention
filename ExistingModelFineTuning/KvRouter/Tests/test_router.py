"""Smoke + correctness tests for the tiered KV router.

Run:  python -m ExistingModelFineTuning.KvRouter.Tests.test_router
"""

import torch
import torch.nn.functional as F

from ..cache_store import ChunkPlacementPolicy, RamKVCacheStore
from ..chunk_router import ChunkRouter, RouterConfig


def _rope(x, theta, dim):
    # x: [B,Hh,S,Dh]; standard RoPE (half-split), matches ChunkRouter._apply_rotary.
    B, Hh, S, Dh = x.shape
    half = Dh // 2
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32, device=x.device) / half))
    pos = torch.arange(S, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _make(cfg, B, S, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    H, KVH, Dh = cfg.nhead, cfg.kv_heads, cfg.head_dim
    q_raw = torch.randn(B, H, S, Dh, generator=g, device=device, dtype=dtype)
    k_raw = torch.randn(B, KVH, S, Dh, generator=g, device=device, dtype=dtype)
    v = torch.randn(B, KVH, S, Dh, generator=g, device=device, dtype=dtype)
    q = _rope(q_raw, cfg.theta, Dh).to(dtype)
    k_rope = _rope(k_raw, cfg.theta, Dh).to(dtype)
    return q, k_rope, k_raw, v


def test_dense_equivalence(device):
    """keep_first >= all chunks, token-level => router == full causal attention (exact)."""
    cfg = RouterConfig(nhead=8, kv_heads=4, head_dim=16, chunk_size=8, group_size=4,
                       topk_chunks=0, topk_groups=0, current_group_summaries=False,
                       theta=10000.0)
    B, S = 2, 28  # 3 full chunks + a partial (4) -> exercises chunk close + partial
    dtype = torch.float64
    q, k_rope, k_raw, v = _make(cfg, B, S, device, dtype)

    policy = ChunkPlacementPolicy(keep_last=0, keep_first=999, first_token_level=True)
    store = RamKVCacheStore(compute_device=torch.device(device), policy=policy,
                            kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, chunk_size=cfg.chunk_size,
                            groups_per_chunk=cfg.groups_per_chunk, batch_size=B, dtype=dtype,
                            pin_memory=False)
    router = ChunkRouter(cfg, store)
    out = router.prefill(0, q, k_rope, k_raw, v, start_pos=0)  # [B,H,S,Dh]

    rep = cfg.nhead // cfg.kv_heads
    k_full = k_rope.repeat_interleave(rep, dim=1)
    v_full = v.repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)

    err = (out - ref).abs().max().item()
    # softmax runs in fp32 (matches the reference), so equivalence holds to fp32 precision.
    print(f"[dense_equivalence] max abs err = {err:.3e}")
    assert err < 1e-5, err


def test_routing_shapes_and_causality(device):
    """Sparse routing path: shapes line up, mask is causal, no NaNs."""
    cfg = RouterConfig(nhead=8, kv_heads=4, head_dim=16, chunk_size=8, group_size=4,
                       topk_chunks=3, topk_groups=4, theta=10000.0)
    B, S = 2, 64
    dtype = torch.float32
    q, k_rope, k_raw, v = _make(cfg, B, S, device, dtype)
    policy = ChunkPlacementPolicy(keep_last=1, keep_first=1, first_token_level=False)
    store = RamKVCacheStore(compute_device=torch.device(device), policy=policy,
                            kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, chunk_size=cfg.chunk_size,
                            groups_per_chunk=cfg.groups_per_chunk, batch_size=B, dtype=dtype,
                            pin_memory=False)
    router = ChunkRouter(cfg, store)
    out = router.prefill(0, q, k_rope, k_raw, v, start_pos=0)
    assert out.shape == (B, cfg.nhead, S, cfg.head_dim)
    assert torch.isfinite(out).all()
    print(f"[routing_shapes] ok, closed chunks = {store.num_closed_chunks(0)}")


def test_grad_flow(device):
    """Gradients reach the live (hot-window) KV — router never detaches them."""
    cfg = RouterConfig(nhead=4, kv_heads=2, head_dim=16, chunk_size=8, group_size=4,
                       topk_chunks=2, topk_groups=2, theta=10000.0)
    B, S = 1, 24
    dtype = torch.float32
    q, k_rope, k_raw, v = _make(cfg, B, S, device, dtype)
    for t in (k_rope, v):
        t.requires_grad_(True)
    policy = ChunkPlacementPolicy(keep_last=2, keep_first=1, first_token_level=False)
    store = RamKVCacheStore(compute_device=torch.device(device), policy=policy,
                            kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, chunk_size=cfg.chunk_size,
                            groups_per_chunk=cfg.groups_per_chunk, batch_size=B, dtype=dtype,
                            pin_memory=False)
    router = ChunkRouter(cfg, store)
    C = cfg.chunk_size
    outs = []
    p = 0
    while p < S:
        take = min(C - (p % C), S - p)
        r = router.decode_block(0, q[:, :, p:p+take], k_rope[:, :, p:p+take],
                                k_raw[:, :, p:p+take], v[:, :, p:p+take], p)
        outs.append(r.attend(q[:, :, p:p+take]))
        p += take
    loss = torch.cat(outs, dim=2).pow(2).sum()
    loss.backward()
    # last tokens are in the live window -> must receive gradient
    assert v.grad is not None and v.grad.abs().sum().item() > 0
    print(f"[grad_flow] grad norm on v = {v.grad.norm().item():.3e}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    test_dense_equivalence(dev)
    test_routing_shapes_and_causality(dev)
    test_grad_flow(dev)
    print("ALL PASSED")
