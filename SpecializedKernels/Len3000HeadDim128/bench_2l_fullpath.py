"""Remeasure true 2L (means + chunk route + attend) vs 1L-64 @ 256 tokens.

The isolated 169 µs 2L number is attend-only. This script times the same
call `vlm_prefill_attention` / `diffusion_cross_attention` actually make.

    /home/vladimir/my-env/bin/python -m SpecializedKernels.Len3000HeadDim128.bench_2l_fullpath
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SpecializedKernels.Len3000HeadDim128 import (
    diffusion_cross_attention as diff_2l,
    vlm_prefill_attention as vlm_2l,
)
from SpecializedKernels.Len3000HeadDim128.kernel import (
    CHUNK as CHUNK_2L,
    GPC,
    GROUP,
    HEAD_DIM,
    MAX_CHUNKS,
    MAX_GROUPS,
    TOPK_C,
    TOPK_G,
    _diff_hga2_kernel,
    _vlm_hga2_kernel,
    chunk_route_diff,
    chunk_route_vlm,
    fill_chunk_keys,
    n_chunks,
)
from SpecializedKernels.Len3000HeadDim128.other_variants.attention import (
    diffusion_cross_attention as diff_1l,
    vlm_prefill_attention as vlm_1l,
)
from SpecializedKernels.Len3000HeadDim128.other_variants.kernel import (
    CHUNK as CHUNK_1L,
    N_BLOCK,
    fill_chunk_keys as fill_chunk_keys_1l,
    n_chunks as n_chunks_1l,
)


def _median_us(fn, warmup: int = 25, iters: int = 80) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2] * 1e3


def _row(name: str, us: float, ref: float | None = None) -> None:
    if ref is None or ref <= 0:
        print(f"  {name:<44} {us:8.1f} µs")
    else:
        print(f"  {name:<44} {us:8.1f} µs   {ref / us:5.2f}× vs 1L-64/256")


def _bench_vlm(s: int, h: int, kvh: int, dtype: torch.dtype) -> None:
    b, d = 1, HEAD_DIM
    scale = d ** -0.5
    q = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
    k = torch.randn(b, kvh, s, d, device="cuda", dtype=dtype)
    v = torch.randn(b, kvh, s, d, device="cuda", dtype=dtype)
    out_2l = torch.empty_like(q)
    out_1l = torch.empty_like(q)
    out_sdpa = torch.empty_like(q)

    gk = torch.empty(b, kvh, MAX_GROUPS, d, device="cuda", dtype=dtype)
    route = torch.empty(b, h, MAX_CHUNKS, TOPK_C, device="cuda", dtype=torch.int32)
    ck_1l = torch.empty(b, kvh, N_BLOCK, d, device="cuda", dtype=dtype)

    nc = n_chunks(s, CHUNK_2L)
    ng = n_chunks(s, GROUP)

    def fill_gk():
        fill_chunk_keys(k, gk, s, chunk=GROUP)

    def route_vlm():
        chunk_route_vlm(q, k, out=route[:, :, :nc])

    def attend_2l():
        _vlm_hga2_kernel[(ng, b * h)](
            q, k, v, gk, route,
            out_2l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            gk.stride(0), gk.stride(1),
            route.stride(0), route.stride(1),
            out_2l.stride(0), out_2l.stride(1), out_2l.stride(2),
            s, ng, h, kvh, float(scale),
            TOPK_G=TOPK_G, TOPK_C=TOPK_C, GPC=GPC, GROUP=GROUP,
            BLOCK_Q=16, BLOCK_D=HEAD_DIM, BLOCK_CAND=64,
            num_warps=4, num_stages=2,
        )

    def full_2l():
        vlm_2l(q, k, v, softmax_scale=scale, out=out_2l, group_k=gk, route=route)

    def fill_ck_1l():
        fill_chunk_keys_1l(k, ck_1l, s, chunk=CHUNK_1L)

    def full_1l():
        # Per-layer VLM: K is new every layer, so 64-token means must be rebuilt.
        # Passing chunk_k only avoids malloc; vlm_1l will not refill same-dtype buffers.
        fill_chunk_keys_1l(k, ck_1l, s, chunk=CHUNK_1L)
        vlm_1l(
            q, k, v, softmax_scale=scale, topk=3, out=out_1l,
            chunk_k=ck_1l, route_block=CHUNK_1L,
        )

    def kernel_1l_only():
        # Means already in ck_1l; kernel still routes (Q vs chunk means) + attends.
        vlm_1l(
            q, k, v, softmax_scale=scale, topk=3, out=out_1l,
            chunk_k=ck_1l, route_block=CHUNK_1L,
        )

    def api_1l_no_buf():
        # Same call as other_variants/bench.py (allocates/fills means internally).
        vlm_1l(q, k, v, softmax_scale=scale, topk=3, route_block=CHUNK_1L)

    def api_2l_no_buf():
        vlm_2l(q, k, v, softmax_scale=scale)

    def sdpa():
        out_sdpa.copy_(
            F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
        )

    # Compile + fill once so component timers see a hot path.
    for _ in range(4):
        full_2l()
        full_1l()
        api_1l_no_buf()
        api_2l_no_buf()
        sdpa()
    torch.cuda.synchronize()

    us_sdpa = _median_us(sdpa)
    us_1l_full = _median_us(full_1l)
    fill_ck_1l()
    us_1l_kern = _median_us(kernel_1l_only)
    us_1l_means = _median_us(fill_ck_1l)
    us_1l_api = _median_us(api_1l_no_buf)

    us_2l_full = _median_us(full_2l)
    us_2l_api = _median_us(api_2l_no_buf)
    us_2l_means = _median_us(fill_gk)
    fill_gk()
    us_2l_route = _median_us(route_vlm)
    fill_gk()
    route_vlm()
    us_2l_attn = _median_us(attend_2l)
    us_2l_sum = us_2l_means + us_2l_route + us_2l_attn

    tok_2l = GROUP + TOPK_G * GROUP
    tok_1l = CHUNK_1L * (3 + 1)
    print(f"VLM prefill  S={s}  Hq={h} Hkv={kvh}  2L tokens={tok_2l}  1L tokens={tok_1l}")
    _row("SDPA causal GQA", us_sdpa)
    _row("1L-64 top-3+cur FULL (means+kernel)", us_1l_full)
    _row("  1L fill 64-token means", us_1l_means)
    _row("  1L kernel (route+attend, means reused)", us_1l_kern)
    _row("1L API no prealloc (old isolated bench)", us_1l_api)
    print()
    _row("2L FULL (means + PyTorch top-4 + attend)", us_2l_full, us_1l_full)
    _row("2L API no prealloc", us_2l_api, us_1l_full)
    _row("  2L fill 16-token group means", us_2l_means)
    _row("  2L exact PyTorch chunk top-4", us_2l_route)
    _row("  2L attend kernel only", us_2l_attn, us_1l_full)
    _row("  2L components summed", us_2l_sum, us_1l_full)
    print(f"  2L full / 1L full = {us_2l_full / us_1l_full:5.2f}× slower")
    print(f"  attend-only / 1L full = {us_2l_attn / us_1l_full:5.2f}×  (the misleading ratio)")
    print()
    return {
        "s": s,
        "sdpa": us_sdpa,
        "1l_full": us_1l_full,
        "1l_means": us_1l_means,
        "1l_kern": us_1l_kern,
        "2l_full": us_2l_full,
        "2l_means": us_2l_means,
        "2l_route": us_2l_route,
        "2l_attn": us_2l_attn,
    }


def _bench_diff(s: int, qd: int, h: int, kvh: int, dtype: torch.dtype) -> None:
    b, d = 1, HEAD_DIM
    scale = d ** -0.5
    q = torch.randn(b, h, qd, d, device="cuda", dtype=dtype)
    k = torch.randn(b, kvh, s + qd, d, device="cuda", dtype=dtype)
    v = torch.randn(b, kvh, s + qd, d, device="cuda", dtype=dtype)
    out_2l = torch.empty_like(q)
    out_1l = torch.empty_like(q)

    gk = torch.empty(b, kvh, MAX_GROUPS, d, device="cuda", dtype=dtype)
    ck_2l = torch.empty(b, kvh, MAX_CHUNKS, d, device="cuda", dtype=dtype)
    route = torch.empty(b, h, TOPK_C, device="cuda", dtype=torch.int32)
    ck_1l = torch.empty(b, kvh, N_BLOCK, d, device="cuda", dtype=dtype)

    fill_chunk_keys(k, ck_2l, s, chunk=CHUNK_2L)
    fill_chunk_keys(k, gk, s, chunk=GROUP)
    fill_chunk_keys_1l(k, ck_1l, s, chunk=CHUNK_1L)
    nc = n_chunks(s, CHUNK_2L)
    ng = n_chunks(s, GROUP)

    def route_d():
        chunk_route_diff(q, ck_2l, nc, out=route)

    def attend_2l():
        _diff_hga2_kernel[(b * h,)](
            q, k, v, gk, route,
            out_2l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            gk.stride(0), gk.stride(1),
            route.stride(0), route.stride(1),
            out_2l.stride(0), out_2l.stride(1), out_2l.stride(2),
            s, ng, qd, s + qd, h, kvh, float(scale),
            TOPK_G=TOPK_G, TOPK_C=TOPK_C, GPC=GPC, GROUP=GROUP,
            BLOCK_Q=64, BLOCK_D=HEAD_DIM, BLOCK_CAND=32,
            num_warps=4, num_stages=2,
        )

    def full_2l_reused():
        # Means reused (ROS Euler loop). PyTorch top-4 still every step.
        diff_2l(
            q, k, v, softmax_scale=scale, n_prompt=s, out=out_2l,
            chunk_k=ck_2l, group_k=gk, route=route,
        )

    def full_2l_recompute():
        diff_2l(q, k, v, softmax_scale=scale, n_prompt=s, out=out_2l)

    def full_1l_reused():
        diff_1l(
            q, k, v, softmax_scale=scale, topk=3, n_prompt=s, out=out_1l,
            chunk_k=ck_1l, route_block=CHUNK_1L,
        )

    def full_1l_recompute():
        diff_1l(
            q, k, v, softmax_scale=scale, topk=3, n_prompt=s, out=out_1l,
            route_block=CHUNK_1L,
        )

    def sdpa():
        F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale, enable_gqa=True)

    for _ in range(4):
        full_2l_reused()
        full_1l_reused()
        sdpa()
    torch.cuda.synchronize()

    us_sdpa = _median_us(sdpa)
    us_1l_re = _median_us(full_1l_reused)
    us_1l_rc = _median_us(full_1l_recompute)
    us_2l_re = _median_us(full_2l_reused)
    us_2l_rc = _median_us(full_2l_recompute)
    us_2l_route = _median_us(route_d)
    route_d()
    us_2l_attn = _median_us(attend_2l)

    print(f"Diffusion  Q={qd}  prefix={s}  means reused unless noted")
    _row("SDPA full 64×(S+64)", us_sdpa)
    _row("1L-64 top-3 reused means (route in kernel)", us_1l_re)
    _row("1L-64 recompute 64-token means", us_1l_rc)
    _row("2L FULL reused means (PyTorch top-4+attend)", us_2l_re, us_1l_re)
    _row("  2L exact PyTorch chunk top-4", us_2l_route)
    _row("  2L attend kernel only", us_2l_attn, us_1l_re)
    _row("2L recompute chunk+group means", us_2l_rc, us_1l_re)
    print(f"  2L reused / 1L reused = {us_2l_re / us_1l_re:5.2f}× slower")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seqlens", type=int, nargs="+", default=[2324, 2688, 3000])
    p.add_argument("--q-diff", type=int, default=64)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"GPU  {torch.cuda.get_device_name(0)}")
    print(f"2L   chunk={CHUNK_2L} group={GROUP} topk_c={TOPK_C} topk_g={TOPK_G} "
          f"tokens={GROUP + TOPK_G * GROUP}")
    print(f"1L   block={CHUNK_1L} topk=3+current tokens={CHUNK_1L * 4}")
    print(f"GQA  Hq={args.heads} Hkv={args.kv_heads} D={HEAD_DIM} bf16")
    print()

    for s in args.seqlens:
        print("=" * 72)
        _bench_vlm(s, args.heads, args.kv_heads, torch.bfloat16)
        _bench_diff(s, args.q_diff, args.heads, args.kv_heads, torch.bfloat16)


if __name__ == "__main__":
    main()
