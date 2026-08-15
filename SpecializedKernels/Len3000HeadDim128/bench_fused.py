"""Correctness + speed of the specialized 128/16 2L kernels vs 1L-64/256.

    /home/vladimir/my-env/bin/python -m SpecializedKernels.Len3000HeadDim128.bench_fused
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SpecializedKernels.Len3000HeadDim128.attention import (
    diffusion_cross_attention,
    vlm_prefill_attention,
)
from SpecializedKernels.Len3000HeadDim128.kernel import (
    CHUNK,
    GROUP,
    HEAD_DIM,
    MAX_CHUNKS,
    MAX_GROUPS,
    TOPK_C,
    chunk_route_vlm,
    chunk_route_vlm_fast,
    fill_hga_means,
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
)


def _median_us(fn, warmup=15, iters=50) -> float:
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


def _check_route(s: int, hq: int, kvh: int) -> None:
    b, d = 1, HEAD_DIM
    q = torch.randn(b, hq, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, kvh, s, d, device="cuda", dtype=torch.bfloat16)
    gk = torch.empty(b, kvh, MAX_GROUPS, d, device="cuda", dtype=torch.bfloat16)
    ck = torch.empty(b, kvh, MAX_CHUNKS, d, device="cuda", dtype=torch.bfloat16)
    fill_hga_means(k, gk, ck, s)
    nc = n_chunks(s, CHUNK)
    rt = torch.empty(b, hq, MAX_CHUNKS, TOPK_C, device="cuda", dtype=torch.int32)
    chunk_route_vlm_fast(q, ck, rt, s)
    ref = chunk_route_vlm(q, k)
    a, b_ = rt[:, :, :nc], ref
    late = slice(4, nc)  # chunks with a full top-4 of previous
    aa, bb = a[:, :, late], b_[:, :, late]
    agree = (aa == bb).all(dim=-1).float().mean().item() if nc > 4 else float("nan")
    flat_a = aa.reshape(-1, TOPK_C)
    flat_b = bb.reshape(-1, TOPK_C)
    match = sum(
        set(flat_a[i].tolist()) == set(flat_b[i].tolist())
        for i in range(flat_a.shape[0])
    )
    tot = max(flat_a.shape[0], 1)
    n_full = s // CHUNK
    kc = k[:, :, : n_full * CHUNK].reshape(1, kvh, n_full, CHUNK, HEAD_DIM).mean(dim=3)
    ck_err = (ck[:, :, :n_full].float() - kc.float()).abs().mean().item() if n_full else 0.0
    print(
        f"  route S={s}: late exact-order {agree*100:.1f}%  "
        f"same-set {100*match/tot:.1f}%  ck L1 {ck_err:.5f}"
    )


def _check_vlm(s: int, hq: int, kvh: int) -> None:
    b, d = 1, HEAD_DIM
    scale = d ** -0.5
    torch.manual_seed(0)
    q = torch.randn(b, hq, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, kvh, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, kvh, s, d, device="cuda", dtype=torch.bfloat16)
    o = vlm_prefill_attention(q, k, v, softmax_scale=scale)
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    l1 = (o.float() - dense.float()).abs().mean().item()
    cos = F.cosine_similarity(o.float().flatten(), dense.float().flatten(), dim=0).item()
    print(f"  vlm  S={s}: L1 vs SDPA {l1:.4f}  cos {cos:.3f}  finite={torch.isfinite(o).all().item()}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    hq, kvh = 32, 8
    print(f"GPU  {torch.cuda.get_device_name(0)}")
    print("correctness")
    for s in (512, 2324, 2688, 3000):
        _check_route(s, hq, kvh)
        _check_vlm(s, hq, kvh)

    print("\nspeed vs 1L-64/256")
    dtype = torch.bfloat16
    d = HEAD_DIM
    scale = d ** -0.5
    for s in (2324, 2688, 3000):
        q = torch.randn(1, hq, s, d, device="cuda", dtype=dtype)
        k = torch.randn(1, kvh, s, d, device="cuda", dtype=dtype)
        v = torch.randn(1, kvh, s, d, device="cuda", dtype=dtype)
        out2 = torch.empty_like(q)
        out1 = torch.empty_like(q)
        gk = torch.empty(1, kvh, MAX_GROUPS, d, device="cuda", dtype=dtype)
        ck = torch.empty(1, kvh, MAX_CHUNKS, d, device="cuda", dtype=dtype)
        rt = torch.empty(1, hq, MAX_CHUNKS, TOPK_C, device="cuda", dtype=torch.int32)
        qc = torch.empty(1, hq, MAX_CHUNKS, d, device="cuda", dtype=dtype)
        ck1 = torch.empty(1, kvh, N_BLOCK, d, device="cuda", dtype=dtype)

        def full_2l():
            vlm_prefill_attention(
                q, k, v, softmax_scale=scale, out=out2,
                group_k=gk, chunk_k=ck, route=rt, q_chunk=qc,
            )

        def full_1l():
            fill_chunk_keys_1l(k, ck1, s, chunk=CHUNK_1L)
            vlm_1l(q, k, v, softmax_scale=scale, topk=3, out=out1, chunk_k=ck1, route_block=CHUNK_1L)

        def sdpa():
            F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)

        for _ in range(3):
            full_2l()
            full_1l()
        us2 = _median_us(full_2l)
        us1 = _median_us(full_1l)
        usd = _median_us(sdpa, warmup=8, iters=20)

        qd = 64
        q_d = torch.randn(1, hq, qd, d, device="cuda", dtype=dtype)
        k_d = torch.randn(1, kvh, s + qd, d, device="cuda", dtype=dtype)
        v_d = torch.randn(1, kvh, s + qd, d, device="cuda", dtype=dtype)
        outd = torch.empty_like(q_d)
        fill_hga_means(k_d, gk, ck, s)
        fill_chunk_keys_1l(k_d, ck1, s, chunk=CHUNK_1L)
        rtd = torch.empty(1, hq, TOPK_C, device="cuda", dtype=torch.int32)

        def d2():
            diffusion_cross_attention(
                q_d, k_d, v_d, softmax_scale=scale, n_prompt=s, out=outd,
                chunk_k=ck, group_k=gk, route=rtd,
            )

        def d1():
            diff_1l(
                q_d, k_d, v_d, softmax_scale=scale, topk=3, n_prompt=s, out=outd,
                chunk_k=ck1, route_block=CHUNK_1L,
            )

        for _ in range(3):
            d2()
            d1()
        ud2 = _median_us(d2)
        ud1 = _median_us(d1)
        print(
            f"  S={s:4d}  VLM 2L {us2:6.1f} µs  1L {us1:6.1f} µs  "
            f"({us2/us1:4.2f}×)  SDPA {usd:6.1f}  |  "
            f"Diff 2L {ud2:6.1f}  1L {ud1:6.1f}  ({ud2/ud1:4.2f}×)"
        )


if __name__ == "__main__":
    main()
