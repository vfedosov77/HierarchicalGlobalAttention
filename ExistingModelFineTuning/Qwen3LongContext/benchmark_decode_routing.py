"""Decode-routing benchmark for the sticky per-token request optimisation.

This benchmarks the ``ChunkRouter`` *attention-only* decode path (no FP8 weights, so it runs on
any CUDA GPU, including SM 8.6 boxes that cannot load the real FP8 30B model).  It builds a
synthetic router + ``VramKVCacheStore`` at the real 30B attention shapes (H=32, KV=4, Dh=128,
chunk=64, group=16), prefills ``--prefill`` tokens, then times ``--decode`` single-token decode
steps for each routing config in a small matrix.

What it measures per config
---------------------------
* **decode tok/s**          — wall-clock decode throughput (router + attend only).
* **kv_gather ms/tok**      — time spent inside ``store.gather_tokens`` (the opened-group token
  fetch — the PCIe-heavy op the optimisation targets), timed with CUDA events.
* **sticky-reuse metrics**  — chunk/group request counts, sticky-reused vs new-added per token,
  and ``group_gather_count`` (new groups fetched/token) — which should *drop* after the first
  few tokens of each active chunk as later tokens reuse earlier tokens' opened groups.
* **out err vs legacy**     — max-abs / mean-cosine of the routed attention output vs the
  ``legacy`` config (decode_sticky off, full per-token request == the pre-optimisation code),
  a cheap proxy for how much the routed *content* changed.

What it does NOT measure (run on the real model)
------------------------------------------------
``ppl``/short-eval loss, **greedy token match vs old code**, and long-context QA quality need the
real Qwen3-30B weights.  Reproduce those by varying the new ``--topk-chunks-request`` /
``--topk-groups-request`` / ``--decode-sticky`` knobs (now plumbed through
``replace_qwen_attention_with_router``) in::

    python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed --bench \
        --layers 4 --tokens 8192 --keep-first 2 --keep-last 8 \
        --topk 20 --topk-groups 32 --group-size 16 --max-new 64

Run this synthetic benchmark::

    python -m ExistingModelFineTuning.Qwen3LongContext.benchmark_decode_routing \
        --prefill 8192 --decode 128
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ExistingModelFineTuning.KvRouter import (
    ChunkPlacementPolicy,
    ChunkRouter,
    RamKVCacheStore,
    RouterConfig,
    VramKVCacheStore,
)


# ---------------------------------------------------------------------------
# Test matrix (label, topk_chunks, topk_chunks_request, topk_groups, topk_groups_request,
#              decode_sticky).  ``request=None`` -> half the working set (the new default).
# ---------------------------------------------------------------------------
def default_matrix() -> List[Tuple[str, int, Optional[int], int, Optional[int], bool]]:
    return [
        # legacy: pre-optimisation behaviour (full per-token request, no sticky reuse) == reference
        ("legacy   ", 20, 20, 32, 16, False),
        # baseline: working set == legacy, but groups requested at 16 with sticky reuse
        ("baseline  ", 20, 20, 32, 16, True),
        ("quality   ", 20, 10, 32, 16, True),
        ("balanced  ", 20, 8, 32, 12, True),
        ("fast      ", 20, 6, 32, 8, True),
    ]


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _rotary_table(theta: float, S: int, Dh: int, device: torch.device, dtype: torch.dtype):
    half = Dh // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(S, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


@dataclass
class Result:
    label: str
    tok_s: float
    kv_ms_per_tok: float
    metrics: Dict[str, int]
    out: torch.Tensor  # [T, H, Dh]
    hit_rate: float = 0.0


def run_config(
    label: str, topk_chunks: int, topk_chunks_request: Optional[int],
    topk_groups: int, topk_groups_request: Optional[int], decode_sticky: bool,
    *, B: int, H: int, KVH: int, Dh: int, C: int, gs: int, theta: float,
    keep_first: int, keep_last: int, prefill: int, decode: int,
    q: torch.Tensor, k_rope: torch.Tensor, k_raw: torch.Tensor, v: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, device: torch.device, dtype: torch.dtype,
    store_kind: str, vram_cache_chunks: int,
) -> Result:
    cfg = RouterConfig(
        nhead=H, kv_heads=KVH, head_dim=Dh, chunk_size=C, group_size=gs,
        topk_chunks=topk_chunks, topk_groups=topk_groups,
        topk_chunks_request=topk_chunks_request, topk_groups_request=topk_groups_request,
        decode_sticky=decode_sticky, theta=theta,
    )
    policy = ChunkPlacementPolicy(keep_last=keep_last, keep_first=keep_first, first_token_level=True)
    if store_kind == "ram":
        # Cold KV in host RAM, routed chunks pulled to a bounded VRAM LRU bank => gather_tokens
        # crosses PCIe only for *newly* required groups (sticky reuse => bank hits => cheap).
        store = RamKVCacheStore(
            compute_device=device, policy=policy, kv_heads=KVH, head_dim=Dh,
            chunk_size=C, groups_per_chunk=C // gs, batch_size=B, dtype=dtype,
            storage_device=torch.device("cpu"), pin_memory=device.type == "cuda",
            vram_cache_chunks=vram_cache_chunks, vram_summary_chunks=8192, num_layers=1,
        )
    else:
        store = VramKVCacheStore(
            compute_device=device, policy=policy, kv_heads=KVH, head_dim=Dh,
            chunk_size=C, groups_per_chunk=C // gs, batch_size=B, dtype=dtype,
        )
    router = ChunkRouter(cfg, store)
    router.reset()

    # --- prefill the resident context (vectorized multi-chunk path; populates the store) ---
    with torch.inference_mode():
        router.route_query_block(
            0, q[:, :, :prefill], k_rope[:, :, :prefill], k_raw[:, :, :prefill], v[:, :, :prefill],
            0, cos=cos[:, :, :prefill], sin=sin[:, :, :prefill],
        )
    # Free the prefill's transient allocations so the LRU VRAM bank can size itself against the
    # real free VRAM (otherwise reserved memory makes the bank tiny -> uncached, artificially slow).
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    # Count cache hits/misses for the decode phase only.
    if hasattr(store, "cache_hits"):
        store.cache_hits = 0
        store.cache_misses = 0

    # Time only the opened-group token fetch (gather_tokens) with CUDA events.
    use_cuda = device.type == "cuda"
    gather_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    orig_gather = store.gather_tokens

    def timed_gather(layer, *a, **k):  # type: ignore[no-untyped-def]
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            r = orig_gather(layer, *a, **k)
            e.record()
            gather_events.append((s, e))
            return r
        return orig_gather(layer, *a, **k)

    store.gather_tokens = timed_gather  # type: ignore[assignment]
    router.collect_metrics = True

    outs: List[torch.Tensor] = []
    with torch.inference_mode():
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for p in range(prefill, prefill + decode):
            segs = router.route_query_block(
                0, q[:, :, p:p + 1], k_rope[:, :, p:p + 1], k_raw[:, :, p:p + 1], v[:, :, p:p + 1], p,
            )
            for routed, lo, hi in segs:
                outs.append(routed.attend(q[:, :, p:p + 1][:, :, lo:hi], use_summaries=False)[0, :, 0])
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    store.gather_tokens = orig_gather  # type: ignore[assignment]
    kv_ms = sum(s.elapsed_time(e) for s, e in gather_events) if use_cuda else 0.0
    out = torch.stack(outs, dim=0).float()  # [T, H, Dh]
    hits = getattr(store, "cache_hits", 0)
    misses = getattr(store, "cache_misses", 0)
    hit_rate = hits / max(1, hits + misses)
    res = Result(label.strip(), decode / dt, kv_ms / decode, dict(router.decode_metrics), out, hit_rate)
    del store, router
    if use_cuda:
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefill", type=int, default=8192, help="resident context tokens (multiple of chunk)")
    ap.add_argument("--decode", type=int, default=128, help="single-token decode steps to time")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=8)
    ap.add_argument("--theta", type=float, default=10_000_000.0)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--store", choices=["ram", "vram"], default="ram",
                    help="ram = host-RAM cold tier (real PCIe gather, bank cache); vram = all resident")
    ap.add_argument("--vram-cache", type=int, default=256, help="LRU VRAM chunk-bank upper bound (ram store)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    B, H, KVH, Dh = 1, args.heads, args.kv_heads, args.head_dim
    C, gs = args.chunk_size, args.group_size
    S = (args.prefill // C) * C
    T = args.decode
    total = S + T

    g = torch.Generator(device=device).manual_seed(args.seed)
    q_raw = torch.randn(B, H, total, Dh, generator=g, device=device, dtype=dtype)
    k_raw = torch.randn(B, KVH, total, Dh, generator=g, device=device, dtype=dtype)
    v = torch.randn(B, KVH, total, Dh, generator=g, device=device, dtype=dtype)
    cos, sin = _rotary_table(args.theta, total, Dh, device, dtype)
    q = _rope(q_raw, cos, sin)
    k_rope = _rope(k_raw, cos, sin)  # cos/sin [1,1,S,Dh] broadcast across the KV heads

    print(f"device={device}  dtype={args.dtype}  H={H} KV={KVH} Dh={Dh} chunk={C} group={gs}")
    print(f"prefill={S} tok ({S // C} chunks)  decode={T} tok  keep_first={args.keep_first} "
          f"keep_last={args.keep_last}\n", flush=True)

    common = dict(
        B=B, H=H, KVH=KVH, Dh=Dh, C=C, gs=gs, theta=args.theta,
        keep_first=args.keep_first, keep_last=args.keep_last, prefill=S, decode=T,
        q=q, k_rope=k_rope, k_raw=k_raw, v=v, cos=cos, sin=sin, device=device, dtype=dtype,
        store_kind=args.store, vram_cache_chunks=args.vram_cache,
    )

    ref: Optional[Result] = None
    header = (f"{'config':10s} {'tok/s':>8s} {'kv_gather':>10s} "
              f"{'chunk req/stk/new':>18s} {'group req/stk/new':>18s} {'g_gather/tok':>12s} "
              f"{'err_vs_legacy':>18s}")
    print(header)
    print("-" * len(header), flush=True)
    for (label, tc, tcr, tg, tgr, sticky) in default_matrix():
        res = run_config(label, tc, tcr, tg, tgr, sticky, **common)
        if ref is None:
            ref = res  # legacy is first -> reference
        m = res.metrics
        calls = max(1, m["calls"])
        bh = max(1, m["bh"])
        per = lambda key: m[key] / calls / bh  # noqa: E731  per-token, per-(batch·head)
        if res.out.shape == ref.out.shape:
            max_err = (res.out - ref.out).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                res.out.flatten(1), ref.out.flatten(1), dim=1).mean().item()
            err_str = f"{max_err:.2e}/{cos_sim:.4f}"
        else:
            err_str = "n/a"
        chunk_str = f"{per('chunk_req_count'):.1f}/{per('chunk_sticky_reused'):.1f}/{per('chunk_new_added'):.1f}"
        group_str = f"{per('group_req_count'):.1f}/{per('group_sticky_reused'):.1f}/{per('group_new_added'):.1f}"
        print(f"{res.label:10s} {res.tok_s:8.1f} {res.kv_ms_per_tok:8.3f}ms "
              f"{chunk_str:>18s} {group_str:>18s} {per('group_gather_count'):12.1f} {err_str:>18s}",
              flush=True)

    print("\nLegend: req/stk/new = per-token requested / sticky-reused / newly-added (per head).")
    print("g_gather/tok = NEW opened groups fetched per token (the routing-level PCIe proxy;")
    print("               bypass-immune, monotonic in budget; lower = more in-chunk sticky reuse).")
    print("kv_gather    = measured CUDA time of the opened-group token fetch (host->VRAM gather).")
    print("err_vs_legacy = max-abs / mean-cosine of attention output vs the legacy config.")


if __name__ == "__main__":
    main()
