"""Self-checks for the segment-TBPTT boundary contract (``commit`` / ``rewind``).

These guard the streaming long-context training path (the author's blocked-inference pattern +
per-segment backward).  Two properties must hold or training silently corrupts:

* **rewind idempotency** — replaying a segment after ``rewind`` reproduces the byte-identical
  routed output, so a gradient-checkpointing recompute re-appends the same chunks into the same
  store slots instead of double-populating;
* **commit stop-gradient** — ``commit`` detaches the live hot-window tensors in place, bounding the
  autograd graph (and VRAM) to a single segment (truncated BPTT).

Run:  python -m ExistingModelFineTuning.KvRouter.Tests.test_segment_tbptt
"""

import torch

from ..cache_store import ChunkPlacementPolicy, RamKVCacheStore
from ..chunk_router import ChunkRouter, RouterConfig


def _rope(x, theta, dim, start=0):
    B, Hh, S, Dh = x.shape
    half = Dh // 2
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32, device=x.device) / half))
    pos = torch.arange(start, start + S, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def _make(cfg, B, S, device, dtype, start=0, seed=0, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    H, KVH, Dh = cfg.nhead, cfg.kv_heads, cfg.head_dim
    q_raw = torch.randn(B, H, S, Dh, generator=g, device=device, dtype=dtype)
    k_raw = torch.randn(B, KVH, S, Dh, generator=g, device=device, dtype=dtype)
    v = torch.randn(B, KVH, S, Dh, generator=g, device=device, dtype=dtype)
    q = _rope(q_raw, cfg.theta, Dh, start).to(dtype)
    k_rope = _rope(k_raw, cfg.theta, Dh, start).to(dtype)
    if requires_grad:
        k_rope.requires_grad_(True); v.requires_grad_(True)
    return q, k_rope, k_raw, v


def _make_store(cfg, B, device, dtype):
    policy = ChunkPlacementPolicy(keep_last=2, keep_first=1, first_token_level=False)
    return RamKVCacheStore(compute_device=torch.device(device), policy=policy,
                           kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, chunk_size=cfg.chunk_size,
                           groups_per_chunk=cfg.groups_per_chunk, batch_size=B, dtype=dtype,
                           pin_memory=False)


def _attend_block(router, q, k_rope, k_raw, v, start_pos, populate_store=True):
    """Route a block and run the attention; return concatenated output [B,H,S,Dh]."""
    segs = router.route_query_block(0, q, k_rope, k_raw, v, start_pos, populate_store=populate_store)
    out = q.new_zeros(q.shape)
    for routed, lo, hi in segs:
        out[:, :, lo:hi] = routed.attend(q[:, :, lo:hi])
    return out


def _cfg():
    return RouterConfig(nhead=4, kv_heads=2, head_dim=16, chunk_size=8, group_size=4,
                        topk_chunks=2, topk_groups=2, theta=10000.0)


def test_rewind_idempotent_vectorized(device):
    """seg-0 seeded by the vectorized prefill path; replaying seg-1 after rewind is identical."""
    cfg = _cfg()
    B, seg_len = 1, 16  # 2 chunks/segment
    dtype = torch.float64
    store = _make_store(cfg, B, device, dtype)
    router = ChunkRouter(cfg, store)

    q0, kr0, kraw0, v0 = _make(cfg, B, seg_len, device, dtype, start=0, seed=1)
    _attend_block(router, q0, kr0, kraw0, v0, start_pos=0)        # seed the store
    router.commit(0)
    n_after_seed = store.num_closed_chunks(0)

    q1, kr1, kraw1, v1 = _make(cfg, B, seg_len, device, dtype, start=seg_len, seed=2)
    out_a = _attend_block(router, q1, kr1, kraw1, v1, start_pos=seg_len)
    router.rewind(0)
    assert store.num_closed_chunks(0) == n_after_seed, "rewind did not restore n_closed"
    out_b = _attend_block(router, q1, kr1, kraw1, v1, start_pos=seg_len)

    err = (out_a - out_b).abs().max().item()
    print(f"[rewind_idempotent_vectorized] max abs err = {err:.3e}")
    assert err == 0.0, err


def test_rewind_idempotent_incremental(device):
    """seg-0 seeded chunk-by-chunk (incremental); replaying seg-1 after rewind is identical."""
    cfg = _cfg()
    B, seg_len = 1, 16
    dtype = torch.float64
    C = cfg.chunk_size
    store = _make_store(cfg, B, device, dtype)
    router = ChunkRouter(cfg, store)

    q0, kr0, kraw0, v0 = _make(cfg, B, seg_len, device, dtype, start=0, seed=3)
    p = 0
    while p < seg_len:                                            # seed via decode_block
        take = min(C - (p % C), seg_len - p)
        router.decode_block(0, q0[:, :, p:p+take], kr0[:, :, p:p+take],
                            kraw0[:, :, p:p+take], v0[:, :, p:p+take], p)
        p += take
    router.commit(0)
    n_after_seed = store.num_closed_chunks(0)

    q1, kr1, kraw1, v1 = _make(cfg, B, seg_len, device, dtype, start=seg_len, seed=4)
    out_a = _attend_block(router, q1, kr1, kraw1, v1, start_pos=seg_len)
    router.rewind(0)
    assert store.num_closed_chunks(0) == n_after_seed, "rewind did not restore n_closed"
    out_b = _attend_block(router, q1, kr1, kraw1, v1, start_pos=seg_len)

    err = (out_a - out_b).abs().max().item()
    print(f"[rewind_idempotent_incremental] max abs err = {err:.3e}")
    assert err == 0.0, err


def test_commit_stops_gradient(device):
    """commit detaches the live hot-window tensors in place (truncated BPTT)."""
    cfg = _cfg()
    B, seg_len = 1, 16
    dtype = torch.float32
    C = cfg.chunk_size
    store = _make_store(cfg, B, device, dtype)
    router = ChunkRouter(cfg, store)

    # incremental seed leaves grad-carrying live windows in the store
    q0, kr0, kraw0, v0 = _make(cfg, B, seg_len, device, dtype, start=0, seed=5, requires_grad=True)
    p = 0
    while p < seg_len:
        take = min(C - (p % C), seg_len - p)
        router.decode_block(0, q0[:, :, p:p+take], kr0[:, :, p:p+take],
                            kraw0[:, :, p:p+take], v0[:, :, p:p+take], p)
        p += take

    st = store._layer(0)
    live_before = any(t.requires_grad for d in (st.live_group_k, st.live_group_v,
                                                st.live_token_k, st.live_token_v) for t in d.values())
    assert live_before, "expected grad-carrying live windows before commit"

    router.commit(0)
    st = store._layer(0)
    for d in (st.live_group_k, st.live_group_v, st.live_token_k, st.live_token_v):
        for t in d.values():
            assert not t.requires_grad, "commit must detach live windows in place"
    for d in (st.committed_live_group_k, st.committed_live_group_v,
              st.committed_live_token_k, st.committed_live_token_v):
        for t in d.values():
            assert not t.requires_grad, "committed snapshot must be detached"
    print("[commit_stops_gradient] live + committed windows detached")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    test_rewind_idempotent_vectorized(dev)
    test_rewind_idempotent_incremental(dev)
    test_commit_stops_gradient(dev)
    print("ALL PASSED")
