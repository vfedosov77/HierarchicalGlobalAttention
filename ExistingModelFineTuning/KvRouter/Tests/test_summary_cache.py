"""Tests for the independent group-summary VRAM cache (the routing accelerator).

These exercise the RAM tier (record on CPU, compute on the GPU) where the cache is active and
prove three things:

1. **Correctness** — ``gather_group_summaries`` served from the bounded LRU summary cache returns
   *exactly* the same values as the direct (cache-off) record gather, for arbitrary routed-chunk
   index tensors, across LRU eviction.
2. **Decoupling** — group routing (``gather_group_summaries``) no longer touches the *token* bank:
   it costs only ``M·Dh`` summary copies, never a whole-chunk (``C·Dh``) download.  Only the chunks
   actually *opened* (``gather_tokens`` / ``gather_chunk_tokens_kvh``) ever enter the token bank.
3. **Token bank still correct** after dropping its group-summary slots.

Run:  python -m ExistingModelFineTuning.KvRouter.Tests.test_summary_cache
"""

from __future__ import annotations

import torch

from ..cache_store import ChunkPlacementPolicy, RamKVCacheStore
from ..chunk_router import ChunkRouter, RouterConfig
from .test_router import _make


def _seed(store, layer, N, B, KVH, M, C, Dh, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    ck = torch.randn(B, KVH, N, Dh, generator=g, device=device, dtype=dtype)
    gk = torch.randn(B, KVH, N, M, Dh, generator=g, device=device, dtype=dtype)
    gv = torch.randn(B, KVH, N, M, Dh, generator=g, device=device, dtype=dtype)
    tk = torch.randn(B, KVH, N, C, Dh, generator=g, device=device, dtype=dtype)
    tv = torch.randn(B, KVH, N, C, Dh, generator=g, device=device, dtype=dtype)
    store.seed_closed_chunks(layer, ck, gk, gv, tk, tv)
    return gk, gv, tk, tv


def _store(device, *, B, KVH, Dh, C, M, dtype, scap, tcap):
    return RamKVCacheStore(
        compute_device=torch.device(device),
        policy=ChunkPlacementPolicy(keep_last=0, keep_first=0, first_token_level=False),
        kv_heads=KVH, head_dim=Dh, chunk_size=C, groups_per_chunk=M, batch_size=B, dtype=dtype,
        pin_memory=False, storage_device=torch.device("cpu"),
        vram_cache_chunks=tcap, vram_summary_chunks=scap, num_layers=1, vram_cache_reserve_gb=0.0,
    )


def test_summary_cache_correctness(device):
    """Cached gather_group_summaries == direct record gather, over LRU eviction (cap < N)."""
    B, H, KVH, Dh, C, M = 1, 8, 2, 16, 8, 2
    N, dtype, rep = 40, torch.float32, 8 // 2
    ref = _store(device, B=B, KVH=KVH, Dh=Dh, C=C, M=M, dtype=dtype, scap=0, tcap=0)   # cache off
    cad = _store(device, B=B, KVH=KVH, Dh=Dh, C=C, M=M, dtype=dtype, scap=12, tcap=0)  # cap < N
    _seed(ref, 0, N, B, KVH, M, C, Dh, dtype, device)
    _seed(cad, 0, N, B, KVH, M, C, Dh, dtype, device)

    g = torch.Generator(device=device).manual_seed(7)
    max_err = 0.0
    for _ in range(30):  # many steps so the cap-12 cache must evict and reload
        K = int(torch.randint(1, 10, (1,), generator=g, device=device).item())
        idx = torch.randint(0, N, (B, H, K), generator=g, device=device)
        rgk, rgv = ref.gather_group_summaries(0, idx)
        cgk, cgv = cad.gather_group_summaries(0, idx)
        max_err = max(max_err, (rgk - cgk).abs().max().item(), (rgv - cgv).abs().max().item())
    assert cad.cache_misses == 0, "group routing must NOT touch the token bank"
    assert cad.summary_misses > 0 and cad.summary_hits > 0
    print(f"[summary_cache_correctness] max abs err = {max_err:.3e}  "
          f"(summary {cad.summary_hits} hit / {cad.summary_misses} miss, token-bank untouched)")
    assert max_err == 0.0, max_err


def test_token_bank_correct(device):
    """gather_chunk_tokens_kvh / gather_tokens still exact after dropping the bank's summary slots."""
    B, KVH, Dh, C, M = 1, 2, 16, 8, 2
    N, dtype = 24, torch.float32
    ref = _store(device, B=B, KVH=KVH, Dh=Dh, C=C, M=M, dtype=dtype, scap=0, tcap=0)
    cad = _store(device, B=B, KVH=KVH, Dh=Dh, C=C, M=M, dtype=dtype, scap=0, tcap=6)
    _, _, tk, tv = _seed(ref, 0, N, B, KVH, M, C, Dh, dtype, device)
    _seed(cad, 0, N, B, KVH, M, C, Dh, dtype, device)

    g = torch.Generator(device=device).manual_seed(3)
    max_err = 0.0
    for _ in range(20):
        idx = torch.randint(0, N, (B, KVH, 4), generator=g, device=device)
        rk, rv = ref.gather_chunk_tokens_kvh(0, idx)
        ck, cv = cad.gather_chunk_tokens_kvh(0, idx)
        max_err = max(max_err, (rk - ck).abs().max().item(), (rv - cv).abs().max().item())
    print(f"[token_bank_correct] max abs err = {max_err:.3e}  "
          f"({cad.cache_hits} hit / {cad.cache_misses} miss)")
    assert max_err == 0.0, max_err


def test_router_decode_decoupled(device):
    """End-to-end: routed-middle churn is absorbed by the summary cache; the token bank only ever
    loads the chunks whose groups are actually opened.  Drives the incremental routing path
    (``prefill`` streams chunk-by-chunk through ``decode_block``) and reports both hit-rates."""
    cfg = RouterConfig(nhead=8, kv_heads=2, head_dim=16, chunk_size=8, group_size=4,
                       topk_chunks=6, topk_groups=4, theta=10000.0)
    B, S, dtype = 1, 8 * 30, torch.float32  # 30 chunks of context
    q, k_rope, k_raw, v = _make(cfg, B, S, device, dtype)
    store = _store(device, B=B, KVH=cfg.kv_heads, Dh=cfg.head_dim, C=cfg.chunk_size,
                   M=cfg.groups_per_chunk, dtype=dtype, scap=4096, tcap=64)
    store.policy = ChunkPlacementPolicy(keep_last=2, keep_first=2, first_token_level=False)
    router = ChunkRouter(cfg, store)

    # prefill() streams the sequence chunk-by-chunk through decode_block, so every routed-middle
    # gather (summary cache) and every opened-group gather (token bank) is exercised.
    router.prefill(0, q, k_rope, k_raw, v, start_pos=0)

    sh, sm = store.summary_hits, store.summary_misses
    th, tm = store.cache_hits, store.cache_misses
    s_hr = 100.0 * sh / max(1, sh + sm)
    t_hr = 100.0 * th / max(1, th + tm)
    print(f"[router_decode_decoupled] summary cache {s_hr:.1f}% hit ({sh}/{sh+sm}); "
          f"token bank {t_hr:.1f}% hit ({th}/{th+tm})")
    # The summary cache spans the whole context → each chunk is loaded at most once (≈0 re-misses).
    assert sm <= 30, f"summary cache should hold the whole context, misses={sm}"
    print("[router_decode_decoupled] ok")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    test_summary_cache_correctness(dev)
    test_token_bank_correct(dev)
    test_router_decode_decoupled(dev)
    print("ALL PASSED")
