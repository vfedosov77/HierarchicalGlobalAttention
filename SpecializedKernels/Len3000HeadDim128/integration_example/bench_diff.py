"""Isolated diffusion: old gather-reuse vs pack-and-attend vs SDPA.

    /home/vladimir/my_env/bin/python \\
        SpecializedKernels/Len3000HeadDim128/integration_example/bench_diff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from SpecializedKernels.Len3000HeadDim128.integration_example.attention import (  # noqa: E402
    diffusion_cross_attention,
)
from SpecializedKernels.Len3000HeadDim128.integration_example.kernel import (  # noqa: E402
    HEAD_DIM,
    N_PACK_PAD,
    fill_hga_means,
)


def median_us(fn, warmup=25, iters=80) -> float:
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
    scale = HEAD_DIM ** -0.5
    s, qd = 2929, 64
    print(f"GPU {torch.cuda.get_device_name(0)}  S={s} Q={qd}")

    for hq, kvh, tag in ((16, 8, "expert 16/8"),):
        print(f"\n=== {tag} ===")
        q = torch.randn(1, hq, qd, HEAD_DIM, device=device, dtype=dtype)
        k = torch.randn(1, kvh, s + qd, HEAD_DIM, device=device, dtype=dtype)
        v = torch.randn_like(k)
        out = torch.empty(1, qd, hq, HEAD_DIM, device=device, dtype=dtype)
        gk = torch.empty(1, kvh, 256, HEAD_DIM, device=device, dtype=dtype)
        ck = torch.empty(1, kvh, 32, HEAD_DIM, device=device, dtype=dtype)
        sel = torch.empty(1, hq, 16, device=device, dtype=torch.int32)
        pk = torch.zeros(1, hq, N_PACK_PAD, HEAD_DIM, device=device, dtype=dtype)
        pv = torch.zeros_like(pk)
        fill_hga_means(k, gk, ck, s)

        def route():
            diffusion_cross_attention(
                q, k, v, softmax_scale=scale, n_prompt=s, out=out,
                chunk_k=ck, group_k=gk, sel=sel, pack_k=pk, pack_v=pv,
                reuse_route=False,
            )

        def reuse():
            diffusion_cross_attention(
                q, k, v, softmax_scale=scale, n_prompt=s, out=out,
                chunk_k=ck, group_k=gk, sel=sel, pack_k=pk, pack_v=pv,
                reuse_route=True,
            )

        def sdpa():
            F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale, enable_gqa=True)

        mask = torch.zeros(1, 1, qd, s + qd, device=device, dtype=torch.float32)

        def sdpa_mask():
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)

        route()
        ref = out.clone()
        reuse()
        l1 = (out.float() - ref.float()).abs().mean().item()
        print(f"  L1 reuse vs route {l1:.5f}  finite={torch.isfinite(out).all().item()}")

        us_r = median_us(route)
        us_u = median_us(reuse)
        us_s = median_us(sdpa)
        us_m = median_us(sdpa_mask)
        print(f"  route+pack+attend  {us_r:7.1f} µs")
        print(f"  reuse packed       {us_u:7.1f} µs")
        print(f"  SDPA unmasked      {us_s:7.1f} µs")
        print(f"  SDPA + 4D zeros    {us_m:7.1f} µs")

        n_steps, n_layers, every = 10, 36, 3
        n_route = (n_steps + every - 1) // every
        n_reuse = n_steps - n_route
        mix = (n_route * us_r + n_reuse * us_u) * n_layers / 1000
        print(f"  10×{n_layers} every-3 pack  {mix:6.2f} ms")
        print(f"  10×{n_layers} always SDPA   {n_steps * n_layers * us_s / 1000:6.2f} ms")
        print(f"  10×{n_layers} SDPA+mask     {n_steps * n_layers * us_m / 1000:6.2f} ms")


if __name__ == "__main__":
    main()
