"""Isolated: full 2L diffusion route vs reuse-attend, and vs SDPA.

    /home/vladimir/my_env/bin/python -m SpecializedKernels.Len3000HeadDim128.bench_route_reuse
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from SpecializedKernels.Len3000HeadDim128.integration_example.attention import (
    diffusion_cross_attention,
)
from SpecializedKernels.Len3000HeadDim128.integration_example.kernel import (
    CHUNK,
    GROUP,
    HEAD_DIM,
    fill_hga_means,
    n_chunks,
)


def median_us(fn, warmup=20, iters=80) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2] * 1e3


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hq, kvh, d = 32, 8, HEAD_DIM
    s, qd = 2929, 64
    scale = d ** -0.5
    print(f"GPU {torch.cuda.get_device_name(0)}  S={s} Q={qd} GQA {hq}/{kvh}")

    q = torch.randn(1, hq, qd, d, device=device, dtype=dtype)
    k = torch.randn(1, kvh, s + qd, d, device=device, dtype=dtype)
    v = torch.randn_like(k)
    out = torch.empty(1, qd, hq, d, device=device, dtype=dtype)
    gk = torch.empty(1, kvh, 256, d, device=device, dtype=dtype)
    ck = torch.empty(1, kvh, 32, d, device=device, dtype=dtype)
    sel = torch.empty(1, hq, 16, device=device, dtype=torch.int32)
    fill_hga_means(k, gk, ck, s)

    def route():
        diffusion_cross_attention(
            q, k, v, softmax_scale=scale, n_prompt=s, out=out,
            chunk_k=ck, group_k=gk, sel=sel, reuse_route=False,
        )

    def reuse():
        diffusion_cross_attention(
            q, k, v, softmax_scale=scale, n_prompt=s, out=out,
            chunk_k=ck, group_k=gk, sel=sel, reuse_route=True,
        )

    def sdpa():
        F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale, enable_gqa=True)

    route()  # populate sel
    print(f"  sel unique groups (head0): {sel[0, 0].tolist()}")

    us_r = median_us(route)
    us_u = median_us(reuse)
    us_s = median_us(sdpa)
    print(f"  route+attend     {us_r:7.1f} µs")
    print(f"  reuse attend     {us_u:7.1f} µs   ({us_r / us_u:.2f}× vs route)" if us_u else "")
    print(f"  SDPA 64×(S+64)   {us_s:7.1f} µs")

    n_steps, n_layers = 10, 36
    every = 3
    n_route = (n_steps + every - 1) // every
    n_reuse = n_steps - n_route
    mix = (n_route * us_r + n_reuse * us_u) * n_layers / 1000
    all_r = n_steps * n_layers * us_r / 1000
    all_s = n_steps * n_layers * us_s / 1000
    print(f"\n  10 steps × 36 layers (attention only):")
    print(f"    always route     {all_r:6.2f} ms")
    print(f"    every-3 + reuse  {mix:6.2f} ms   save {all_r - mix:5.2f} ms")
    print(f"    always SDPA      {all_s:6.2f} ms")


if __name__ == "__main__":
    main()
